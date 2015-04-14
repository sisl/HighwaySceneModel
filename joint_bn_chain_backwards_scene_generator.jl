
type JointBNChainBackwardsSceneGenerator <: SceneGenerator
    net        :: Network
    discmap    :: Dict{Symbol, AbstractDiscretizer}
    varindeces :: Dict{Symbol, Cint}
    ordering   :: Vector{Int}
end

function train_jointbnchainbackwardsscenegenerator(scenes::Vector{RoadScene}, nbins::Dict{Symbol, Int},
    bounds :: Dict{Symbol, (Float64, Float64)} = [
        :speed        => BOUNDS_V,
        :d_cl         => BOUNDS_D_CL,
        :yaw          => BOUNDS_YAW,
        :d_front      => BOUNDS_D_FRONT,
        :v_front      => BOUNDS_V,
        :d_rear       => BOUNDS_D_FRONT,
        :v_rear       => BOUNDS_V]
    )

    symbols = collect(keys(nbins))
    nvars = length(symbols)
    nscenes = length(scenes)

    ncars = 0
    for scene in scenes
        ncars += length(scene.vehicles)
    end
    
    varindeces = Dict{Symbol,Cint}()
    discmap = Dict{Symbol, AbstractDiscretizer}()
    r_arr = Array(Int, nvars)
    for (i,sym) in enumerate(keys(bounds))
        @assert(in(sym, symbols))
        varindeces[sym] = int32(i-1)

        nlabels = nbins[sym]
        bins = linspace(bounds[sym][1], bounds[sym][2], nlabels+1)
        discmap[sym] = LinearDiscretizer(bins, Cint)

        r_arr[i] = nlabels
    end

    dmat = Array(Int, nvars, ncars) # NOTE(tim): dmat contains class labels > 0 (ie, 1,2,3,4...)
    add_to_dmat(sym::Symbol, count::Int, val::Float64) = 
        dmat[varindeces[sym]+1, count] = encode(discmap[sym], val)

    count = 0
    for scene in scenes

        vehicles = scene.vehicles
        sceneinfo = scene.info

        for (i,veh) in enumerate(vehicles)

            count += 1

            info = sceneinfo[i]
            lane = info.laneindex
            
            add_to_dmat(:d_cl,          count, info.d_cl)
            add_to_dmat(:yaw,           count, veh.ϕ)
            add_to_dmat(:speed,         count, veh.v)

            if info.carind_fore != 0
                d_front = max(calc_d_front(scene, i), 0.0)
                v_front = scene.vehicles[info.carind_fore].v
                add_to_dmat(:d_front, count, d_front)
                add_to_dmat(:v_front, count, v_front)
            else
                count -= 1
                continue
            end

            if info.carind_rear != 0
                d_rear = max(calc_d_rear(scene, i), 0.0)
                v_rear = scene.vehicles[info.carind_rear].v
                add_to_dmat(:d_rear, count, d_rear)
                add_to_dmat(:v_rear, count, v_rear)
            else
                count -= 1
                continue
            end
        end
    end
    dmat = dmat[:,1:count]

    # structure learning
    params = LearnParams_BayesianSearch()
    params.maxparents  = 5
    params.niterations = 20
    net, worked = learn((dmat-1)', params)
    @assert(worked)

    # add a dirichlet prior and ensure that
    # there are no non-zero probabilities
    net2 = create_new_net_with_prior_counts(net, r_arr, dmat)
    set_default_BN_algorithm(net2, DSL_ALG_BN_LAURITZEN)

    ordering = to_native_int_array(partial_ordering(net)::IntArray)
    JointBNChainBackwardsSceneGenerator(net2, discmap, varindeces, ordering)
end
function generate_scene(
    sg   :: JointBNChainBackwardsSceneGenerator,
    road :: StraightRoadway
    )
    
    net = sg.net
    discmap = sg.discmap
    varindeces = sg.varindeces

    nlanes = road.nlanes
    centers = lanecenters(road)

    max_n_vehicles = int(nlanes*roadlength(road) / 10.0) # NOTE(tim): quick and dirty estimate of upper limit
    vehicles = Array(Vehicle, max_n_vehicles)
    vehiclecount = 0

    for lane = 1 : nlanes
        
        # generate the first vehicle

        center = centers[lane]
        assignment = rand(net, ordering=sg.ordering)::Dict{Cint, Cint}

        d_rear    = decode(discmap[:d_rear],  int32(assignment[varindeces[:d_rear ]]+1))
        d_front   = decode(discmap[:d_front], int32(assignment[varindeces[:d_front]]+1))
        v_rear    = decode(discmap[:v_rear],  int32(assignment[varindeces[:v_rear ]]+1))
        v         = decode(discmap[:speed],   int32(assignment[varindeces[:speed  ]]+1))
        d_cl      = decode(discmap[:d_cl],    int32(assignment[varindeces[:d_cl   ]]+1))
        yaw       = decode(discmap[:yaw],     int32(assignment[varindeces[:yaw    ]]+1))

        y = road.horizon_fore - d_front*rand()
        vehicles[vehiclecount+=1] = Vehicle(center+d_cl,y,yaw,v)

        # now generate vehicles out rearward
        while y - d_front - VEHICLE_MEAN_LENGTH > road.horizon_rear

            y -= (d_front + VEHICLE_MEAN_LENGTH)

            empty!(assignment)
            assignment[varindeces[:d_front]] = encode(discmap[:d_front], d_rear)-1
            assignment[varindeces[:v_front]] = encode(discmap[:v_front],      v)-1
            assignment[varindeces[:speed]]   = encode(discmap[:speed],   v_rear)-1

            rand!(net, assignment, ordering=sg.ordering)

            v       = v_rear
            d_rear  = decode(discmap[:d_rear],  int32(assignment[varindeces[:d_rear]]+1))
            v_rear  = decode(discmap[:v_rear],  int32(assignment[varindeces[:v_rear]]+1))
            d_cl    = decode(discmap[:d_cl],    int32(assignment[varindeces[:d_cl   ]]+1))
            yaw     = decode(discmap[:yaw],     int32(assignment[varindeces[:yaw    ]]+1))
            vehicles[vehiclecount+=1] = Vehicle(center+d_cl,y,yaw,v)
        end
    end

    scene = RoadScene(road, vehicles[1:vehiclecount])
    calc_vehicle_extra_info!(scene)
    scene
end
function loglikelihood(
    sg    :: JointBNChainBackwardsSceneGenerator,
    scene :: RoadScene
    )
    
    score = 0.0

    net = sg.net
    discmap = sg.discmap
    varindeces = sg.varindeces
    ordering = sg.ordering

    vehicles  = scene.vehicles
    sceneinfo = scene.info

    nodeid2sym = Dict{Cint, Symbol}()
    for (sym, bmap) in sg.discmap
        nodeid2sym[varindeces[sym]] = sym
    end

    assignment = (Cint=>Cint)[]
    evidence   = (Cint=>Cint)[]

    f_assign(d::Dict{Cint,Cint}, sym::Symbol, val::Float64) =
        d[varindeces[sym]] = encode(discmap[sym], val)-1

    bmap_dfront = discmap[:d_front]
    bmap_drear  = discmap[:d_rear]

    node_dfront = varindeces[:d_front]
    node_drear  = varindeces[:d_rear]
    node_speed  = varindeces[:speed]

    binwidths_dfront = binwidths(bmap_dfront)
    binwidths_drear = binwidths(bmap_drear)

    for (carind,veh) in enumerate(scene.vehicles)

        info = scene.info[carind]

        has_car_front = info.carind_fore != 0
        has_car_rear  = info.carind_rear != 0

        if has_car_front && has_car_rear
            # middle of lane
            # P(dcl,ϕ,drear,vrear|dfront,vfront,v)

            f_assign(evidence,   :d_front,  calc_d_front(scene, carind))
            f_assign(evidence,   :v_front,  scene.vehicles[info.carind_fore].v)
            f_assign(evidence,   :speed,   veh.v)
            
            delete!(assignment, node_speed)
            f_assign(assignment, :d_cl,    info.d_cl)
            f_assign(assignment, :yaw,     veh.ϕ)
            f_assign(assignment, :d_rear, calc_d_rear(scene, carind))
            f_assign(assignment, :v_rear, scene.vehicles[info.carind_rear].v)

            probvec = probabilities(net, assignment, evidence, ordering=ordering)
            for (nodeid, prob) in probvec
                w  = binwidth(discmap[nodeid2sym[nodeid]], int32(assignment[nodeid]+1))
                score += log(prob / w)
                @assert(!isinf(score))
            end

        elseif has_car_rear

            f_assign(assignment, :speed,   veh.v)
            f_assign(assignment, :d_cl,    info.d_cl)
            f_assign(assignment, :yaw,     veh.ϕ)
            f_assign(assignment, :d_rear,  calc_d_rear(scene, carind))
            f_assign(assignment, :v_rear,  scene.vehicles[info.carind_rear].v) 

            probvec = probabilities(net, assignment, ordering=ordering)
            for (nodeid, prob) in probvec
                w  = binwidth(discmap[nodeid2sym[nodeid]], int32(assignment[nodeid]+1))
                score += log(prob / w)
                @assert(!isinf(score))
            end

            offset_y = scene.road.horizon_fore - veh.y
            probvec_dfront = normalize!(get_marginal_probability_vec(net, node_dfront, assignment)) ./ binwidths_dfront
            score += log(prob_gen(offset_y, Inf, bmap_dfront.binedges, probvec_dfront ))
            @assert(!isinf(score))

        elseif has_car_front

            f_assign(evidence,   :d_front,  calc_d_front(scene, carind))
            f_assign(evidence,   :v_front,  scene.vehicles[info.carind_fore].v)
            f_assign(evidence,   :speed,    veh.v)

            empty!(assignment)
            f_assign(assignment, :d_cl,    info.d_cl)
            f_assign(assignment, :yaw,     veh.ϕ)

            probvec = probabilities(net, assignment, evidence, ordering=ordering)
            for (nodeid, prob) in probvec
                w  = binwidth(discmap[nodeid2sym[nodeid]], int32(assignment[nodeid]+1))
                score += log(prob / w)
                @assert(!isinf(score))
            end            

            offset_end = scene.road.horizon_rear + veh.y - VEHICLE_MEAN_LENGTH
            if offset_end > 0.0
                probvec_drear = normalize!(get_marginal_probability_vec(net, node_drear, merge(assignment, evidence))) ./ binwidths_drear
                score += log(prob_gt(offset_end, bmap_drear.binedges, probvec_drear))
                @assert(!isinf(score))
            end
        else

            empty!(assignment)
            f_assign(assignment, :d_cl,  info.d_cl)
            f_assign(assignment, :yaw,   veh.ϕ)
            f_assign(assignment, :speed, veh.v)

            probvec = probabilities(net, assignment, ordering=ordering)
            for (nodeid, prob) in probvec
                w  = binwidth(discmap[nodeid2sym[nodeid]], int32(assignment[nodeid]+1))
                score += log(prob / w)
                @assert(!isinf(score))
            end

            offset_y = scene.road.horizon_fore - veh.y
            probvec_dfront = normalize!(get_marginal_probability_vec(net, node_dfront, assignment)) ./ binwidths_dfront
            score += log(prob_gen(offset_y, Inf, bmap_dfront.binedges, probvec_dfront ))
            @assert(!isinf(score))

            offset_end = scene.road.horizon_rear + veh.y - VEHICLE_MEAN_LENGTH
            if offset_end > 0.0
                probvec_drear = normalize!(get_marginal_probability_vec(net, node_drear, merge(assignment, evidence))) ./ binwidths_drear
                score += log(prob_gt(offset_end, bmap_drear.binedges, probvec_drear))
                @assert(!isinf(score))
            end
        end

        @assert(!isinf(score))
    end

    score
end

train{T<:JointBNChainBackwardsSceneGenerator}(res::ModelOptimizationResults{T}, scenes::Vector{RoadScene}) =
    train_jointbnchainbackwardsscenegenerator(scenes, convert(Dict{Symbol,Int}, res.params))

function cross_validate(
    T::Type{JointBNChainBackwardsSceneGenerator},
    scenes::Vector{Trajdata_.RoadScene},
    nbins::Dict{Symbol,Int};
    CV_nfolds :: Int = 10, # k in k-fold cross validations
    CV_rounds :: Int = 10  # number of CV evals to do
    )

    scores = Trajdata_.cross_validate_scenegenerator(T, scenes, nbins = nbins, CV_rounds = CV_rounds, CV_nfolds =CV_nfolds)
    μ, σ = Trajdata_.ave_crossvalidated_likelihood(scores)
    (μ, σ)
end
function cross_validate_scenegenerator(
    ::Type{JointBNChainBackwardsSceneGenerator},
    scenes::Vector{RoadScene};
    CV_nfolds :: Int = 10, # k in k-fold cross validations
    CV_rounds :: Int = 10, # number of CV evals to do
    nbins     :: Dict{Symbol, Int} =  [
                    :speed    => 5,
                    :d_cl     => 5,
                    :yaw      => 5,
                    :d_front  => 5,
                    :d_rear   => 5,
                    :v_front  => 5,
                    :v_rear   => 5,
                    ]
    )
    
    nscenes = length(scenes)

    f_train(inds::Vector{Int}) = train_jointbnchainbackwardsscenegenerator(scenes[inds], nbins)
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
            # print("set!")
            test_inds = setdiff(1:nscenes, train_inds)
            # println(" [DONE]")
            # print("training!")
            model = f_train(train_inds)
            # println(" [DONE]")
            # print("logl!")
            scores[fold,round] = f_test(model, test_inds)
            # println(" [DONE]")
        end
    end

    # println("Done with a round !")

    scores
end
function cyclic_coordinate_ascent_parallel(
    T::Type{JointBNChainBackwardsSceneGenerator}, 
    scenes::Vector{Trajdata_.RoadScene};
    CV_nfolds :: Int = 10,
    CV_rounds :: Int = 10
    )

    param_options = (
            [5,7,10,15,20,25], # :speed
            [5,7,10,15,20,25], # :d_cl
            [5,7,10,15,20,25], # :yaw
            [5,7,10,15,20,25], # :d_front
            [5,7,10,15,20,25], # :d_rear
            [5,7,10,15,20,25], # :v_front
            [5,7,10,15,20,25], # :v_rear
        )

    n_params = length(param_options)
    paraminds = ones(Int, n_params)
    for i = 1 : n_params
        if length(param_options[i]) > 1
            paraminds[i] = int(length(param_options[i])/2)
        end
    end
    params_tried = Set{Vector{Int}}()

    converged = false
    best_score = -Inf
    best_μ     = NaN
    best_σ     = NaN
    iter = 0
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
                    cross_validate(T, scenes,
                       [
                        :speed    => param_options[1][newparams[1]],
                        :d_cl     => param_options[2][newparams[2]],
                        :yaw      => param_options[3][newparams[3]],
                        :d_front  => param_options[4][newparams[4]],
                        :d_rear   => param_options[5][newparams[5]],
                        :v_front  => param_options[6][newparams[6]],
                        :v_rear   => param_options[7][newparams[7]],
                        ],CV_rounds = CV_rounds, CV_nfolds =CV_nfolds)
                end
                toc()

                
                run_ind = 0
                run_score = -Inf
                run_μ = NaN
                run_σ = NaN
                for i = 1 : length(scores)
                    μ = scores[i][1]
                    σ = scores[i][2]
                    score = μ
                    if score > run_score
                        run_ind, run_μ, run_σ, run_score = i, μ, σ, score
                    end
                end

                if run_score > best_score
                    best_score = run_score
                    best_μ, best_σ = run_μ, run_σ
                    paraminds[coordinate] = ip_arr[run_ind]
                    converged = false
                    println("\tfound better: ", coordinate, " -> ", param_options[coordinate][ip_arr[run_ind]])
                    println("\t\tscore: ", best_score)
                end
            end
        end
    end

    println("done!")
    println("optimal params: ", )
    println("\tnbins_speed    = ", param_options[1][paraminds[1]])
    println("\tnbins_d_cl     = ", param_options[2][paraminds[2]])
    println("\tnbins_yaw      = ", param_options[3][paraminds[3]])
    println("\tnbins_d_front  = ", param_options[4][paraminds[4]])
    println("\tnbins_d_rear   = ", param_options[5][paraminds[5]])
    println("\tnbins_v_front  = ", param_options[6][paraminds[6]])
    println("\tnbins_v_rear   = ", param_options[7][paraminds[7]])
    println("score: ", best_score)
    println("iter:  ", iter)

    opt_params = Dict{Symbol,Any}()
    opt_params[:speed  ] = param_options[1][paraminds[1]]
    opt_params[:d_cl   ] = param_options[2][paraminds[2]]
    opt_params[:yaw    ] = param_options[3][paraminds[3]]
    opt_params[:d_front] = param_options[4][paraminds[4]]
    opt_params[:d_rear ] = param_options[5][paraminds[5]]
    opt_params[:v_front] = param_options[6][paraminds[6]]
    opt_params[:v_rear ] = param_options[7][paraminds[7]]

    ModelOptimizationResults{T}(opt_params, best_μ, best_σ, best_score, iter, CV_nfolds, CV_rounds)
end