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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/FieldDataInterface.h"
#include "common/Vector.h"
#include "exec/expression/CompareExpr.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/Expr.h"
#include "exec/expression/function/FunctionFactory.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {
class PhyCallExpr : public Expr {
 public:
    PhyCallExpr(const std::vector<std::shared_ptr<Expr>>& input,
                const std::shared_ptr<const milvus::expr::CallTypeExpr>& expr,
                const std::string& name,
                const segcore::SegmentInternalInterface* segment,
                int64_t active_count,
                int64_t batch_size)
        : Expr(DataType::BOOL, std::move(input), name),
          expr_(expr),
          active_count_(active_count),
          segment_(segment),
          batch_size_(batch_size) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

    void
    MoveCursor() override {
        // TODO: redundant with PhyCompareFilterExpr::MoveCursor, refactor them into a common place
        int64_t processed_rows = 0;
        for (int64_t chunk_id = current_chunk_id_; chunk_id < num_chunk_;
             ++chunk_id) {
            auto chunk_size = chunk_id == num_chunk_ - 1
                                  ? active_count_ - chunk_id * size_per_chunk_
                                  : size_per_chunk_;

            for (int i = chunk_id == current_chunk_id_ ? current_chunk_pos_ : 0;
                 i < chunk_size;
                 ++i) {
                if (++processed_rows >= batch_size_) {
                    current_chunk_id_ = chunk_id;
                    current_chunk_pos_ = i + 1;
                }
            }
        }
    }

 private:
    // TODO: this function is redundant with PhyCompareFilterExpr::GetChunkData,
    // refactor them into a common place
    template <typename T>
    ChunkDataAccessor
    GetChunkData(FieldId field_id, int chunk_id, int data_barrier);

    ChunkDataAccessor
    GetChunkData(DataType data_type,
                 FieldId field_id,
                 int chunk_id,
                 int data_barrier);

    int64_t
    GetNextBatchSize() {
        // TODO: this function is redundant with PhyCompareFilterExpr::GetNextBatchSize,
        // refactor them into a common place
        auto current_rows =
            segment_->type() == SegmentType::Growing
                ? current_chunk_id_ * size_per_chunk_ + current_chunk_pos_
                : current_chunk_pos_;
        return current_rows + batch_size_ >= active_count_
                   ? active_count_ - current_rows
                   : batch_size_;
    }

    VectorPtr
    ExecCallExprDispatcherForHybridSegment(
        expression::FilterScalarFunctionPtr function);

    std::shared_ptr<const milvus::expr::CallTypeExpr> expr_;

    int64_t active_count_{0};
    int64_t num_chunk_{0};
    int64_t current_chunk_id_{0};
    int64_t current_chunk_pos_{0};
    int64_t size_per_chunk_{0};

    const segcore::SegmentInternalInterface* segment_;
    int64_t batch_size_;
};

}  // namespace exec
}  // namespace milvus
