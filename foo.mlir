#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @foo(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2_i32 : i32

    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<2xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<2xi32, #blocked>
    %5 = tt.splat %arg1 : i32 -> tensor<2xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<2xi32, #blocked>

    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x!tt.ptr<bf16>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<2x!tt.ptr<bf16>, #blocked>, tensor<2xi32, #blocked>
    %9 = tt.load %8, %6 : tensor<2x!tt.ptr<bf16>, #blocked>

    %10 = arith.addf %9, %9 fastmath<fast> : tensor<2xbf16, #blocked>

    tt.return
  }
}
