# Filtering Functions

Milvus supports filtering functions in [filtered search](https://milvus.io/docs/single-vector-search.md#Filtered-search).
The function's input parameters can be a column or a literal, and the return type must be bool.

To add a new filtering function, you need to do 2 things:

1) Implement the function.
2) Register the function.

## How to Implement the Function?

The code for filtering function implementation is placed in the `impl` folder.
Taking [starts_with(VARCHAR, VARCHAR)](./impl/StartsWith.cpp) as an example,
`starts_with(strs, prefixes)` returns true if `strs` starts with `prefixes`.

To create the `starts_with(VARCHAR, VARCHAR)` function, define a cpp function `StartsWithVarchar` first:

```cpp
void
StartsWithVarchar(const RowVector& args, FilterFunctionReturn& result) {
    ...
}
```

`starts_with(VARCHAR, VARCHAR)` should only accept two function arguments, so first check if the length of `args` is 2.

```cpp
void
StartsWithVarchar(const RowVector& args, FilterFunctionReturn& result) {
    if (args.childrens().size() != 2) {
        PanicInfo(ExprInvalid,
                  "invalid argument count, expect 2, actual {}",
                  args.childrens().size());
    }
    ...
}
```

Each column's type is actually `SimpleVector`, so the columns need to be converted to `SimpleVector`, and then check if each column's type is `VARCHAR` or `STRING`.

```cpp
void
StartsWithVarchar(const RowVector& args, FilterFunctionReturn& result) {
    ...
    auto strs = std::dynamic_pointer_cast<SimpleVector>(args.child(0));
    Assert(strs != nullptr);
    CheckVarcharOrStringType(strs);
    auto prefixes = std::dynamic_pointer_cast<SimpleVector>(args.child(1));
    Assert(prefixes != nullptr);
    CheckVarcharOrStringType(prefixes);
    ...
}
```

Iterate over each row to compute the result: if either `str` or `prefix` is NULL, then the final result is NULL and `valid_bitmap` is set to false. You can call the `ValidAt` method to check for NULL, and use the `RawValueAt` method to retrieve the raw data.

The result is a `ColumnVector` that has a `bitmap` and a `valid_bitmap`.

```cpp
void
StartsWithVarchar(const RowVector& args, FilterFunctionReturn& result) {
    ...
    for (size_t i = 0; i < strs->size(); ++i) {
        if (strs->ValidAt(i) && prefixes->ValidAt(i)) {
            auto* str_ptr = reinterpret_cast<std::string*>(
                strs->RawValueAt(i, sizeof(std::string)));
            auto* prefix_ptr = reinterpret_cast<std::string*>(
                prefixes->RawValueAt(i, sizeof(std::string)));
            bitmap.set(i, str_ptr->find(*prefix_ptr) == 0);
        } else {
            valid_bitmap[i] = false;
        }
    }
    result = std::make_shared<ColumnVector>(std::move(bitmap),
                                            std::move(valid_bitmap));
    ...
}
```

## How to Register the Function?

In [FunctionFactory.cpp](./FunctionFactory.cpp), call `RegisterFilterFunction` inside `FunctionFactory::RegisterAllFunctions()` method to register the function.

```cpp
void
FunctionFactory::RegisterAllFunctions() {
    RegisterFilterFunction(
        "empty", {DataType::VARCHAR}, function::EmptyVarchar);
    RegisterFilterFunction("starts_with",
                           {DataType::VARCHAR, DataType::VARCHAR},
                           function::StartsWithVarchar);
    ...
}
```
