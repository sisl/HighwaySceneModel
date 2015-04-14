include("piecewise_uniform.jl")

type UnivariateSceneGenerator <: SceneGenerator
    d_front :: ContinuousUnivariateDistribution
    v         :: ContinuousUnivariateDistribution
    d_cl      :: ContinuousUnivariateDistribution
    yaw       :: ContinuousUnivariateDistribution
end

function prob_gt(y::Float64, edges::Vector{Float64}, pdfs::Vector{Float64})
    # compute p(x > y) = ∫ pdf(x) dx from x = y to ∞

    a = y
    b = edges[end]

    if isapprox(a,b)
        return 0.0
    end
    @assert(b > a)

    retval = 0.0
    for i = 1 : length(edges)-1
        lo, hi = edges[i], edges[i+1]
        if a ≤ lo ≤ hi ≤ b # completely contained
            retval += pdfs[i] * (hi-lo)
        elseif lo ≤ a ≤ b ≤ hi # completely surrounds
            retval += pdfs[i] * (b-a)
            return retval
        elseif lo ≤ a ≤ hi # encapsulates a
            retval += pdfs[i] * (hi-a)
        elseif lo ≤ b ≤ hi # encapsulated b
            retval += pdfs[i] * (b-lo)
            return retval
        end
    end

    retval
end
function prob_gen(a::Float64, b::Float64, edges::Vector{Float64}, pdfs::Vector{Float64})
    # compute p(x) = ∫ pdf(u)/u du from u = a to b

    #=
    generative model:
        u ∼ f
        x ∼ U(0,u)
    
    p(x) = ∫f(u)p(x|u)du
         = ∫f(u)/u du from x to ∞

    p(x < y) = ∫∫f(u)/u du dx from x to ∞, from 0 to y
    =#

    if isapprox(a,b)
        return 0.0
    end
    @assert(b > a)

    retval = 0.0
    for i = 1 : length(edges)-1
        lo, hi = edges[i], edges[i+1]
        if a ≤ lo ≤ hi ≤ b # completely contained
            retval += pdfs[i] * log(hi/lo)
        elseif lo ≤ a ≤ b ≤ hi # completely surrounds
            retval += pdfs[i] * log(b/a)
            return retval
        elseif lo ≤ a ≤ hi # encapsulates a
            retval += pdfs[i] * log(hi/a)
        elseif lo ≤ b ≤ hi # encapsulated b
            retval += pdfs[i] * log(b/lo)
            return retval
        end
    end

    retval
end
function prob_uniform_sample_less_than(
    y       :: Float64,
    edges   :: Vector{Float64},
    pdfs    :: Vector{Float64};
    n_steps :: Int = 99
    )

    a = edges[1]
    c = edges[end]

    if y ≥ c
        return 1.0
    end
    b = y

    retval = prob_gen(b,c, edges, pdfs) # + prob_gen(a,c, edges, pdfs) NOTE(tim): this last part is 0

    if mod(n_steps,2) == 0
        n_steps +- 1
    end

    h = (b-a)/n_steps
    odd = true
    x = a
    for i = 1 : n_steps-1
        x += h
        retval += prob_gen(b,c,edges, pdfs) * (2+2*odd)
        odd = !odd
    end
    retval *= h/3

    return retval
end
prob_gt(d::PiecewiseUniform, y::Float64) = prob_gt(y, d.disc.binedges,d.pdfs)
prob_gen(d::PiecewiseUniform, a::Float64, b::Float64) = prob_gen(a, b, d.disc.binedges,d.pdfs)
prob_uniform_sample_less_than(d::PiecewiseUniform, y::Float64; n_steps :: Int = 99) = 
    prob_uniform_sample_less_than(y, d.disc.binedges, d.pdfs, n_steps=n_steps)

function generate_scene(
    sg   :: UnivariateSceneGenerator,
    road :: StraightRoadway
    )
    
    nlanes = road.nlanes
    centers = lanecenters(road)

    max_n_vehicles = int(nlanes*roadlength(road) / 10.0) # NOTE(tim): quick and dirty estimate of upper limit
    vehicles = Array(Vehicle, max_n_vehicles)
    vehiclecount = 0

    for lane = 1 : nlanes
        
        lane_center = centers[lane]
        offset_y = rand()*rand(sg.d_front) - road.horizon_rear
        v        = rand(sg.v)
        y        = offset_y
        vehicles[vehiclecount+=1] = Vehicle(lane_center+rand(sg.d_cl),y,0.0,v)

        # now generate vehicles out forward
        done = false
        while !done
            
            d_front = rand(sg.d_front)

            if y + d_front+VEHICLE_MEAN_LENGTH < road.horizon_fore
                y += d_front+VEHICLE_MEAN_LENGTH
                vehicles[vehiclecount+=1] = Vehicle(lane_center+rand(sg.d_cl),y,rand(sg.yaw),rand(sg.v))
            else
                done = true
            end
        end
    end

    RoadScene(road, vehicles[1:vehiclecount])
end
function loglikelihood(
    sg    :: UnivariateSceneGenerator,
    scene :: RoadScene
    )

    vehicles = scene.vehicles
    if isempty(scene.info)
        calc_vehicle_extra_info!(scene)
    end
    sceneinfo = scene.info

    score = 0.0

    for (i,veh) in enumerate(vehicles)
        info = sceneinfo[i]

        score += logpdf(sg.d_cl, clamp(info.d_cl, -5.75, 5.75))
        score += logpdf(sg.yaw,  clamp(veh.ϕ, -0.1, 0.1))
        score += logpdf(sg.v,    veh.v)

        if info.carind_fore != 0
            d_front = max(calc_d_front(scene, i), 0.0)
            score += logpdf(sg.d_front, d_front)
        else
            # incorporate score for offset_end
            # P(d_front > offset_end_observed)
            offset_end = scene.road.horizon_fore - veh.y - VEHICLE_MEAN_LENGTH
            if offset_end > 0.0
                score += log(prob_gt(sg.d_front, offset_end))
            end
        end

        if info.carind_rear == 0
            # incorporate score for offset_y 
            # P(offset_y)
            offset_y = veh.y + scene.road.horizon_rear
            @assert(offset_y > 0.0)
            score += log(prob_gen(sg.d_front, offset_y, Inf))
        end
    end

    @assert(!isinf(score))

    score
end
function train_univariatescenegenerator_piecewise_uniform(scenes::Vector{RoadScene}, 
    nbins_v        :: Int,
    nbins_d_front  :: Int,
    nbins_d_cl     :: Int,
    nbins_yaw      :: Int
    )
    
    (arr_v, arr_d_front, arr_d_cl, arr_yaw) = get_base_arrays(scenes)

    distr_v       = fit_mle(PiecewiseUniform, arr_v,       nbins_v,       bounds = BOUNDS_V,       laplace_smoothing_counts=1)
    distr_d_front = fit_mle(PiecewiseUniform, arr_d_front, nbins_d_front, bounds = BOUNDS_D_FRONT, laplace_smoothing_counts=1)
    distr_d_cl    = fit_mle(PiecewiseUniform, arr_d_cl,    nbins_d_cl,    bounds = BOUNDS_D_CL,    laplace_smoothing_counts=1)
    distr_yaw     = fit_mle(PiecewiseUniform, arr_yaw,     nbins_yaw,     bounds = BOUNDS_YAW,     laplace_smoothing_counts=1)

    UnivariateSceneGenerator(distr_d_front, distr_v, distr_d_cl, distr_yaw)
end

function train{T<:UnivariateSceneGenerator}(res::ModelOptimizationResults{T}, scenes::Vector{RoadScene})
    p = res.params
    nbins_v         = p[:nbins_v]::Int
    nbins_d_front = p[:nbins_d_front]::Int
    nbins_d_cl      = p[:nbins_d_cl]::Int
    nbins_yaw     = p[:nbins_yaw]::Int
    train_univariatescenegenerator_piecewise_uniform(scenes, nbins_v, nbins_d_front, nbins_d_cl, nbins_yaw)
end

function cross_validate_scenegenerator(
    ::Type{UnivariateSceneGenerator},
    scenes::Vector{RoadScene};
    CV_nfolds :: Int = 10, # k in k-fold cross validations
    CV_rounds :: Int = 10, # number of CV evals to do
    nbins_v         :: Int = 20,
    nbins_d_front :: Int = 20,
    nbins_d_cl      :: Int = 20,
    nbins_yaw     :: Int = 20,
    )
    
    nscenes = length(scenes)

    (arr_v, arr_d_front, arr_d_cl, arr_yaw) = get_base_arrays(scenes)

    f_train(inds::Vector{Int}) = train_univariatescenegenerator_piecewise_uniform(scenes[inds],
        nbins_v, nbins_d_front, nbins_d_cl, nbins_yaw
        )
    f_test(sg::SceneGenerator, inds::Vector{Int}) = begin
        score = 0.0
        for ind in inds
            score += loglikelihood(sg, scenes[ind])
        end
        score
    end

    scores = Array(Float64, CV_nfolds, CV_rounds)
    for round = 1 : CV_rounds
        for (fold, train_inds) in enumerate(Kfold(nscenes, CV_nfolds))
            test_inds = setdiff(1:nscenes, train_inds)
            model = f_train(train_inds)
            scores[fold,round] = f_test(model, test_inds)
        end
    end

    scores
end

function cross_validate(
    ::Type{UnivariateSceneGenerator},
    scenes::Vector{Trajdata_.RoadScene}, 
    nbins::Dict{Symbol,Int};
    CV_nfolds :: Int = 10, # k in k-fold cross validations
    CV_rounds :: Int = 10  # number of CV evals to do
    )

    scores = Trajdata_.cross_validate_scenegenerator(UnivariateSceneGenerator, scenes, CV_rounds = CV_rounds, CV_nfolds =CV_nfolds,
        nbins_v = nbins[:speed],
        nbins_d_front = nbins[:d_front],
        nbins_d_cl = nbins[:d_cl],
        nbins_yaw = nbins[:yaw]
        )
    Trajdata_.ave_crossvalidated_likelihood(scores)
end
function cyclic_coordinate_ascent_parallel(
    T         :: Type{UnivariateSceneGenerator},
    scenes    :: Vector{Trajdata_.RoadScene};
    CV_nfolds :: Int = 10,
    CV_rounds :: Int = 10
    )

    param_options = (
            [2,5,10,50,100,250,500,750,1000,2500,5000], # :speed       
            [2,5,10,50,100,250,500,750,1000,2500,5000], # :d_cl        
            [2,5,10,50,100,250,500,750,1000,2500,5000], # :yaw         
            [2,5,10,50,100,250,500,750,1000,2500,5000], # :d_front 
        )

    n_params = length(param_options)
    paraminds = ones(Int, n_params)
    params_tried = Set{Vector{Int}}()

    converged = false
    best_μ    = -Inf
    best_σ    = NaN
    iter      = 0
    while !converged
        iter += 1
        println("iteration: ", iter)
        
        converged = true
        for coordinate = 1 : n_params
            println("\tcoordinate: ", coordinate)

            to_try = Vector{Int}[]
            ip_arr = Int[]
            for ip in 1 : length(param_options[coordinate])

                newparams = copy(paraminds)
                newparams[coordinate] = ip

                if !in(newparams, params_tried)
                    push!(params_tried, newparams)
                    push!(ip_arr, ip)
                    push!(to_try, newparams)
                end
            end

            if !isempty(to_try)

                tic()
                scores = pmap(to_try) do newparams
                    cross_validate(UnivariateSceneGenerator, scenes,
                       [
                        :speed      => param_options[1][newparams[1]],
                        :d_cl       => param_options[2][newparams[2]],
                        :yaw        => param_options[3][newparams[3]],
                        :d_front  => param_options[4][newparams[4]],
                        ],CV_rounds = CV_rounds, CV_nfolds =CV_nfolds)
                end
                toc()

                run_ind = 0
                run_μ = -Inf
                run_σ = NaN
                for i = 1 : length(scores)
                    μ = scores[i][1]
                    σ = scores[i][2]
                    if μ > run_μ
                        run_ind, run_μ, run_σ = i, μ, σ
                    end
                end

                if run_μ > best_μ
                    best_μ = run_μ
                    best_σ = run_σ
                    paraminds[coordinate] = ip_arr[run_ind]
                    converged = false
                    println("\tfound better: ", coordinate, " -> ", param_options[coordinate][ip_arr[run_ind]])
                    println("\t\tscore: ", best_μ)
                end
            end
        end
    end

    println("done!")
    println("optimal params: ", )
    println("\tnbins_speed     = ", param_options[1][paraminds[1]])
    println("\tnbins_d_cl      = ", param_options[2][paraminds[2]])
    println("\tnbins_yaw       = ", param_options[3][paraminds[3]])
    println("\tnbins_d_front = ", param_options[4][paraminds[4]])
    println("score mean: ", best_μ)
    println("score std:  ", best_σ)
    println("num iter:   ", iter)

    opt_params = Dict{Symbol,Any}()
    opt_params[:nbins_v        ] = param_options[1][paraminds[1]]
    opt_params[:nbins_d_front] = param_options[2][paraminds[2]]
    opt_params[:nbins_d_cl     ] = param_options[3][paraminds[3]]
    opt_params[:nbins_yaw      ] = param_options[4][paraminds[4]]

    ModelOptimizationResults{T}(opt_params, best_μ, best_σ, best_μ, iter, CV_nfolds, CV_rounds)
end