// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/FieldDataInterface.h"
#include "common/Vector.h"
#include "exec/expression/CallExpr.h"
#include "exec/expression/CompareExpr.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/function/FunctionFactory.h"

#include <utility>
#include <vector>

namespace milvus {
namespace exec {

template <typename T>
ChunkDataAccessor
PhyCallExpr::GetChunkData(FieldId field_id, int chunk_id, int data_barrier) {
    if (chunk_id >= data_barrier) {
        auto& indexing = segment_->chunk_scalar_index<T>(field_id, chunk_id);
        if (indexing.HasRawData()) {
            return [&indexing](int i) -> const number {
                return indexing.Reverse_Lookup(i);
            };
        }
    }
    auto chunk_data = segment_->chunk_data<T>(field_id, chunk_id).data();
    return [chunk_data](int i) -> const number { return chunk_data[i]; };
}

template <>
ChunkDataAccessor
PhyCallExpr::GetChunkData<std::string>(FieldId field_id,
                                       int chunk_id,
                                       int data_barrier) {
    if (chunk_id >= data_barrier) {
        auto& indexing =
            segment_->chunk_scalar_index<std::string>(field_id, chunk_id);
        if (indexing.HasRawData()) {
            return [&indexing](int i) -> const std::string {
                return indexing.Reverse_Lookup(i);
            };
        }
    }
    if (segment_->type() == SegmentType::Growing &&
        !storage::MmapManager::GetInstance()
             .GetMmapConfig()
             .growing_enable_mmap) {
        auto chunk_data =
            segment_->chunk_data<std::string>(field_id, chunk_id).data();
        return [chunk_data](int i) -> const number { return chunk_data[i]; };
    } else {
        auto chunk_data =
            segment_->chunk_view<std::string_view>(field_id, chunk_id)
                .first.data();
        return [chunk_data](int i) -> const number {
            return std::string(chunk_data[i]);
        };
    }
}

ChunkDataAccessor
PhyCallExpr::GetChunkData(DataType data_type,
                          FieldId field_id,
                          int chunk_id,
                          int data_barrier) {
    switch (data_type) {
        case DataType::BOOL:
            return GetChunkData<bool>(field_id, chunk_id, data_barrier);
        case DataType::INT8:
            return GetChunkData<int8_t>(field_id, chunk_id, data_barrier);
        case DataType::INT16:
            return GetChunkData<int16_t>(field_id, chunk_id, data_barrier);
        case DataType::INT32:
            return GetChunkData<int32_t>(field_id, chunk_id, data_barrier);
        case DataType::INT64:
            return GetChunkData<int64_t>(field_id, chunk_id, data_barrier);
        case DataType::FLOAT:
            return GetChunkData<float>(field_id, chunk_id, data_barrier);
        case DataType::DOUBLE:
            return GetChunkData<double>(field_id, chunk_id, data_barrier);
        case DataType::VARCHAR: {
            return GetChunkData<std::string>(field_id, chunk_id, data_barrier);
        }
        default:
            PanicInfo(DataTypeInvalid, "unsupported data type: {}", data_type);
    }
}

void
PhyCallExpr::Eval(EvalCtx& context, VectorPtr& result) {
    // NOTE: if the called function is not found, the error should be reported in parser
    std::vector<DataType> fun_param_type_list;
    for (auto& input : this->inputs_) {
        fun_param_type_list.push_back(input->type());
    }
    auto& factory = expression::FunctionFactory::instance();
    auto function =
        factory.getFilterScalarFunction(expression::FilterFunctionRegisterKey{
            this->expr_->fun_name(), std::move(fun_param_type_list)});

    // Currently, we only support function parameters to be columns or constants,
    // so we do not eval the parameters here.
    // TODO: consider nested calls like `func1(func2(a))`, `func(a+b+c, d+e)` in the future

    // TODO:
    // implement ExecCompareExprDispatcherForAllDataSegment:
    // when all the fields has no index, use SIMD for speed up.
    // See also: PhyCompareFilterExpr
    result = ExecCallExprDispatcherForHybridSegment(function);
}

VectorPtr
PhyCallExpr::ExecCallExprDispatcherForHybridSegment(
    expression::FilterScalarFunctionPtr function) {
    auto real_batch_size = GetNextBatchSize();
    if (real_batch_size == 0) {
        return nullptr;
    }

    auto res_vec =
        std::make_shared<ColumnVector>(TargetBitmap(real_batch_size));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);

    std::vector<int64_t> data_barriers;
    for (auto& field_id : expr_->parameter_field_ids_) {
        data_barriers.push_back(segment_->num_chunk_data(field_id));
    }

    int64_t processed_rows = 0;
    for (int64_t chunk_id = current_chunk_id_; chunk_id < num_chunk_;
         ++chunk_id) {
        auto chunk_size = chunk_id == num_chunk_ - 1
                              ? active_count_ - chunk_id * size_per_chunk_
                              : size_per_chunk_;
        std::vector<ChunkDataAccessor> accessors;
        for (int i = 0; i < inputs_.size(); ++i) {
            accessors.push_back(GetChunkData(expr_->parameter_data_types_[i],
                                             expr_->parameter_field_ids_[i],
                                             chunk_id,
                                             data_barriers[i]));
        }

        for (int i = chunk_id == current_chunk_id_ ? current_chunk_pos_ : 0;
             i < chunk_size;
             ++i) {
            std::vector<number> args;
            for (auto& accessor : accessors) {
                args.push_back(accessor(i));
            }
            res[processed_rows++] = function(args);
            if (processed_rows >= batch_size_) {
                current_chunk_id_ = chunk_id;
                current_chunk_pos_ = i + 1;
                return res_vec;
            }
        }
        return res_vec;
    }
}
}  // namespace exec
}  // namespace milvus
