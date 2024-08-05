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

#include "exec/expression/function/RegisterFunction.h"

#include <unordered_map>

namespace milvus {
namespace exec {
namespace expression {

class FunctionFactory {
 public:
    static FunctionFactory&
    instance();

    void
    registerFunction(struct FunctionRegisterKey key,
                     FunctionRegisterFunctionPtr func);

    const FunctionRegisterFunctionPtr
    getFunction(const FunctionRegisterKey& func_sig) const;

 private:
    const FunctionRegisterFunctionPtr
    tryGetFunction(const FunctionRegisterKey& func_sig) const;

    std::unordered_map<FunctionRegisterKey,
                       FunctionRegisterFunctionPtr,
                       FunctionRegisterKey::Hash>
        function_map_;
};

}  // namespace expression
}  // namespace exec
}  // namespace milvus
