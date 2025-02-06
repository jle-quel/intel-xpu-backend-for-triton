module {
  tt.func public @add_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1024_i32_0 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32_0 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<bf16>>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<bf16>>
    %13 = arith.addf %9, %12 : tensor<1024xbf16>
    %14 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
