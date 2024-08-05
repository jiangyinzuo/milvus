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

#include "exec/expression/function/FunctionFactory.h"
#include "exec/expression/function/RegisterFunction.h"

namespace milvus {
namespace exec {
namespace expression {

FunctionFactory&
FunctionFactory::instance() {
    static FunctionFactory factory;
    return factory;
}

void
FunctionFactory::registerFunction(struct FunctionRegisterKey key,
                                  FunctionRegisterFunctionPtr func) {
    function_map_[key] = func;
}

const FunctionRegisterFunctionPtr
FunctionFactory::getFunction(const FunctionRegisterKey& func_sig) const {
    auto func = tryGetFunction(func_sig);
    if (func == nullptr) {
        throw std::runtime_error("Function " + func_sig.toString() +
                                 " not found");
    }
    return func;
}

const FunctionRegisterFunctionPtr
FunctionFactory::tryGetFunction(const FunctionRegisterKey& func_sig) const {
    auto iter = function_map_.find(func_sig);
    if (iter != function_map_.end()) {
        return iter->second;
    }
    return nullptr;
}

}  // namespace expression
}  // namespace exec
}  // namespace milvus
