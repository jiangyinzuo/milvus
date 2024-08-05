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

#include <boost/container_hash/hash.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/Types.h"

namespace milvus {
namespace exec {
namespace expression {

class FunctionFactory;

using FunctionRegisterFunctionPtr =
    void (*)(::milvus::exec::expression::FunctionFactory&);

struct FunctionRegisterKey {
    std::string_view func_name;
    std::vector<DataType> func_param_type_list;

    std::string toString() const;

    bool
    operator==(const FunctionRegisterKey& other) const {
        return func_name == other.func_name &&
               func_param_type_list == other.func_param_type_list;
    }

    struct Hash {
        size_t
        operator()(const FunctionRegisterKey& s) const {
            size_t h1 = std::hash<std::string_view>{}(s.func_name);
            size_t h2 = boost::hash_range(s.func_param_type_list.begin(),
                                          s.func_param_type_list.end());
            return h1 ^ h2;
        }
    };
};

struct FunctionRegisterMap
    : public std::unordered_map<FunctionRegisterKey,
                                FunctionRegisterFunctionPtr,
                                FunctionRegisterKey::Hash> {
    static FunctionRegisterMap&
    instance();
};

struct FunctionRegister {
    FunctionRegister(FunctionRegisterKey key,
                     FunctionRegisterFunctionPtr func_ptr) {
        FunctionRegisterMap::instance().emplace(std::move(key), func_ptr);
    }
};

}  // namespace expression
}  // namespace exec
}  // namespace milvus

#define REGISTER_FUNCTION_IMPL(fn, func_name, register_name)              \
    void func_name(::milvus::exec::expression::FunctionFactory& factory); \
    static ::milvus::exec::expression::FunctionRegister register_name(    \
        #fn, func_name);                                                  \
    void func_name(::milvus::exec::expression::FunctionFactory& factory)

// Define a static object of struct FunctionRegister to register the function.
#define REGISTER_FUNCTION(fn) \
    REGISTER_FUNCTION_IMPL(fn, registerFunction##fn, REGISTER_FUNCTION_##fn)
