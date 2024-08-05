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
#include "exec/expression/EvalCtx.h"
#include "exec/expression/Expr.h"
#include "expr/ITypeExpr.h"

namespace milvus {
namespace exec {

class PhyCallExpr : public Expr {
 public:
    PhyCallExpr(const std::vector<std::shared_ptr<Expr>>& input,
                const std::shared_ptr<const milvus::expr::CallTypeExpr>& expr,
                const std::string& name)
        : Expr(DataType::BOOL, std::move(input), name), expr_(expr) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override;

    void
    MoveCursor() override {
        for (auto& input : inputs_) {
            input->MoveCursor();
        }
    }

 private:
    std::shared_ptr<const milvus::expr::CallTypeExpr> expr_;
};

}  // namespace exec
}  // namespace milvus
