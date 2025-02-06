// -----// IR Dump Before ConvertTritonIntelGPUToLLVM (convert-triton-intel-gpu-to-llvm) ('builtin.module' operation) //----- //

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>

    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<bf16>, #blocked>

    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<bf16>, #blocked>

    %13 = arith.addf %9, %12 fastmath<fast> : tensor<1024xbf16, #blocked>

    %14 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<bf16>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
