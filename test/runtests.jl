using Test
include(joinpath(@__DIR__, "..", "experiment", "PathConfig.jl"))
using .PathConfig

@testset "release configuration" begin
    @test PathConfig.NUM_RUNS > 0
    @test 1 <= PathConfig.NUM_WEEKS <= 52
    @test length(PathConfig.REGIONS) == 4
    @test isapprox(sum(values(PathConfig.WEIGHTS)), 1.0)
    @test isapprox(sum(values(PathConfig.REGION_WEIGHTS)), 1.0)
end
