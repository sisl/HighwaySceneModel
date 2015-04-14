
const OUTPUT_FOLDER = "/media/tim/DATAPART1/Data/NGSIM/output/"

model_name(::Type{UnivariateSceneGenerator})            = "marginalbase"
model_name(::Type{JointBNSimpleSceneGenerator})         = "jointbase"
model_name(::Type{JointBNSceneGenerator})               = "jointlane"
model_name(::Type{JointBNChainSceneGenerator})          = "jointchain"
model_name(::Type{JointBNChainBackwardsSceneGenerator}) = "jointchainbackwards"
model_name(::Type{HeirarchicalSceneGenerator})          = "heirarchical"

function write{T}(res::ModelOptimizationResults{T}, set::String = "")

    if isempty(set)
        filename = joinpath(OUTPUT_FOLDER, model_name(T) * "_res.txt" )
    else
        filename = joinpath(OUTPUT_FOLDER, model_name(T) * "_" * set * "_res.txt" )
    end

    open(filename, "w") do fout

        println(fout, "LIKELIHOOD: ", res.logl_mean, " +- ", res.logl_stdev)
        println(fout, "SCORE:      ", res.score)
        println(fout, "ITER        ", res.iter)
        println(fout, "CV_NFOLDS   ", res.CV_nfolds)
        println(fout, "CV_ROUNDS   ", res.CV_rounds)
        println(fout, "")

        println(fout, "PARAMS")
        println(fout, "======")
        for (k,v) in res.params
            println(fout, k, ": ", v)
        end
    end
end

function write{T<:SceneGenerator}(sg::T, metrics::Dict{Symbol, Any}, set::String = "")
    if isempty(set)
        return write(metrics, model_name(T))
    else
        return write(metrics, model_name(T) * "_" * set)
    end
end
function write(metrics::Dict{Symbol, Any}, set::String = "")

    if isempty(set)
        filename = joinpath(OUTPUT_FOLDER, "metrics.txt" )
    else
        filename = joinpath(OUTPUT_FOLDER, set * "_metrics.txt" )
    end 

    open(filename, "w") do fout
        println(fout, "METRICS")
        println(fout, "=======")
        for (k,v) in metrics
            if isa(v, StreamStats.Var)
                println(fout, k, ": ", mean(v), " +- ", sqrt(v.v_hat))
            elseif isa(v, Vector{Float64})
                m = mean(v)
                println(fout, k, ": ", m, " +- ", stdm(v, m))
                print_tikz_histogram_line(fout, METRIC_DISCS[k], v)
                println(fout, "")
            else
                println(fout, k, ": ", v)
            end
        end
    end
end

function output_scenes{T<:SceneGenerator}(sg::T, scenes::Vector{RoadScene}, set::String = "")

    if isempty(set)
        filename = joinpath(OUTPUT_FOLDER, model_name(T) * "_scene" )
    else
        filename = joinpath(OUTPUT_FOLDER, model_name(T) * "_" * set * "_scene" )
    end

    rm = RenderModel()
    canvas_width = 800
    canvas_height = 200
    set_background_color([1.0,1.0,1.0,0.0], rendermodel=rm)

    for (i,scene) in enumerate(scenes)
        outputfile = @sprintf("%s%d.pdf", filename, i)

        s = CairoPDFSurface(outputfile, canvas_width, canvas_height)
        ctx = creategc(s)

        clear_setup(rendermodel=rm)
        render_scene!(rm, scene)
        center_camera_on_roadway(rm, scene.road)
        render(ctx, canvas_width, canvas_height, rendermodel=rm)
    end
end

function get_histogram_line(disc::LinearDiscretizer, arr::Vector{Float64})
    counts = zeros(Int, nlabels(disc))
    for x in filter!(x->!isnan(x), arr)
        counts[encode(disc, x)] += 1
    end
    zip(bincenters(disc), counts ./ sum(counts))
end
function get_histogram_line(sym::Symbol, arr::Vector{Float64})
    set = get_histogram_line(METRIC_DISCS[sym], arr)
    centers = Array(Float64, length(set))
    probs  = Array(Float64, length(set))
    for (i,tup) in enumerate(set)
        centers[i] = tup[1]
        probs[i] = tup[2]
    end
    (centers, probs)
end
function print_tikz_histogram_line(io::IOStream, disc::LinearDiscretizer, arr::Vector{Float64})

    # this function returns the "() () ()" stuff
    # the first coordinate is the bin center
    # coordinates { (0,10) (20,40) (40,60) }

    for (center,count) in get_histogram_line(disc, arr)
        @printf(io, "(%.4f,%d) ", center, count)
    end
end