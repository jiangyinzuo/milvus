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

#include "exec/expression/function/impl/Empty.h"

#include <boost/variant/get.hpp>
#include <string>
#include "common/EasyAssert.h"
#include "exec/expression/function/FunctionFactory.h"

namespace milvus {
namespace exec {
namespace expression {
namespace function {

bool
EmptyVarchar(const std::vector<FilterFunctionParameter>& args) {
    if (const std::string* string_value = boost::get<std::string>(&args[0])) {
        return string_value->empty();
    }
    PanicInfo(ErrorCode::ExprInvalid, "EmptyStr only accept VARCHAR type.");
}

}  // namespace function
}  // namespace expression
}  // namespace exec
}  // namespace milvus
