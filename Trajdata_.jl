module Trajdata_

using DataFrames
using Distributions
using Discretizers
using Dierckx
using MLBase
using HDF5, JLD

import Base: show, get
import Distributions: loglikelihood

export Trajdata

# vehicle ID <INT>
# frameind <INT> 10 Hz
# n_frames_in_dataset
# epoch_time <INT>
# local_x <FLOAT> front, center of vehicle w/ respect to left-most edge of section
# local_y <FLOAT> front, center of vehicle w/ respect to entry edge of section
# global_x (ft)
# global_y (ft)
# length (ft)
# width (ft)
# class [INT] (1-motorcycle, 2-auto, 3-truck)
# velocity (ft/s²)
# lane (1=farthest left, 5 is farthest right, 6 is aux, 7 is onramp, 8 is offramp)
# carind_front, (0 otherwise)
# carind_rear, (0 otherwise)
# dist_headway_front [ft]
# time_headway_front (9999.9 means v = 0)

const FILENAME_HW_101_0750 = "/media/tim/DATAPART1/Data/NGSIM/HW101/US-101-Main-Data/vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.txt"
const FILENAME_HW_101_0805 = "/media/tim/DATAPART1/Data/NGSIM/HW101/US-101-Main-Data/vehicle-trajectory-data/0805am-0820am/trajectories-0805am-0820am.txt"
const FILENAME_HW_101_0820 = "/media/tim/DATAPART1/Data/NGSIM/HW101/US-101-Main-Data/vehicle-trajectory-data/0820am-0835am/trajectories-0820am-0835am.txt"

const FILENAME_HW_80_0400  = "/media/tim/DATAPART1/Data/NGSIM/HW80/I-80-Main-Data/vehicle-trajectory-data/0400pm-0415pm/trajectories-0400-0415.txt"
const FILENAME_HW_80_0500  = "/media/tim/DATAPART1/Data/NGSIM/HW80/I-80-Main-Data/vehicle-trajectory-data/0500pm-0515pm/trajectories-0500-0515.txt"
const FILENAME_HW_80_0515  = "/media/tim/DATAPART1/Data/NGSIM/HW80/I-80-Main-Data/vehicle-trajectory-data/0515pm-0530pm/trajectories-0515-0530.txt"

const DIST_HW_101_A =  578.0
const DIST_HW_101_B =  698.0
const DIST_HW_101_C =  824.0
const DIST_HW_80_A  =  420.0
const DIST_HW_80_B  = 1230.0

const CLASS_MOTORCYCLE = 1
const CLASS_AUTOMOBILE = 2
const CLASS_TRUCKORBUS = 3

const VEHICLE_MEAN_LENGTH = 14.25 # [ft]
const VEHICLE_MEAN_WIDTH  =  6.15 # [ft]

const AVE_LANE_WIDTH = 11.44 # [ft]
const MAX_ABS_D_CL   =  6.75 # [ft]

const BOUNDS_V       = (0.0,100.0)  # [m/s]
const BOUNDS_D_FRONT = (0.0,300.0)  # [m]
const BOUNDS_D_CL    = (-5.75,5.75) # [m]
const BOUNDS_YAW     = (-0.1,0.1)   # [rad]

const TIMESTEP = 0.1 # 10 Hz

type Trajdata
    df         :: DataFrame
    car2start  :: Dict{Int, Int}         # maps carindex to starting index in the df
    frame2cars :: Dict{Int, Vector{Int}} # maps frameindex to list of carids in the scene

    function Trajdata(input_path::String)

        @assert(isfile(input_path))

        df = readtable(input_path, separator=' ', header = false)
        col_names = [:id, :frameind, :n_frames_in_dataset, :epoch, :local_x, :local_y, :global_x, :global_y, :length, :width, :class, :speed, :acc, :lane, :carind_front, :carind_rear, :dist_headway, :time_headway]
        for (i,name) in enumerate(col_names)
            rename!(df, symbol(@sprintf("x%d", i)), name)
        end

        car2start = Dict{Int, Int}()
        frame2cars = Dict{Int, Vector{Int}}()

        for (dfind, carid) in enumerate(df[:id])
            if !haskey(car2start, carid)
                car2start[carid] = dfind
            end

            frameind = int(df[dfind, :frameind])
            if !haskey(frame2cars, frameind)
                frame2cars[frameind] = [carid]
            else
                frame2cars[frameind] = push!(frame2cars[frameind], carid)
            end
        end

        new(df, car2start, frame2cars)
    end
end
type Vehicle
    x :: Float64
    y :: Float64
    ϕ :: Float64
    v :: Float64

    class  :: Int # ∈ (1-motorcycle, 2-auto, 3-truck)
    length :: Float64
    width  :: Float64

    Vehicle() = new()
    function Vehicle(x::Float64, y::Float64, ϕ::Float64=0.0, v::Float64=0.0;
        class  :: Int     = CLASS_AUTOMOBILE,
        length :: Float64 = VEHICLE_MEAN_LENGTH,
        width  :: Float64 = VEHICLE_MEAN_WIDTH
        )
        new(x, y, ϕ, v, class, length, width)
    end
end
type VehicleExtraInfo
    laneindex   :: Int # what lane it is assigned to
    laneorder   :: Int # what index within the lane it is, increasing in longitudinal direction
    d_cl        :: Float64
    carind_fore :: Int
    carind_rear :: Int
    fake_x      :: Float64

    VehicleExtraInfo() = new()
    VehicleExtraInfo(laneindex::Int, laneorder::Int, d_cl::Float64, carind_fore::Int, carind_rear::Int) = 
        new(laneindex, laneorder, d_cl, carind_fore, carind_rear, 0.0)
end
type StraightRoadway
    # NOTE(tim): all values should be positive
    nlanes       :: Int
    lanewidth    :: Float64
    horizon_fore :: Float64
    horizon_rear :: Float64

    function StraightRoadway(
        nlanes       :: Int     =   5, 
        lanewidth    :: Float64 =  AVE_LANE_WIDTH, 
        horizon_fore :: Float64 = 200.0,
        horizon_rear :: Float64 = 100.0
        )
        @assert(nlanes > 0)
        @assert(lanewidth > 0)
        @assert(horizon_fore > 0)
        @assert(horizon_rear > 0)
        new(nlanes, lanewidth, horizon_fore, horizon_rear)
    end
end
type RoadScene
    road     :: StraightRoadway
    vehicles :: Vector{Vehicle}
    info     :: Vector{VehicleExtraInfo}

    lanespeeds   :: Vector{Float64}
    lanedensities :: Vector{Float64}

    RoadScene(road::StraightRoadway, vehicles::Vector{Vehicle}, info::Vector{VehicleExtraInfo} = VehicleExtraInfo[]) =
        new(road, vehicles, info, Float64[], Float64[])
end
type HighwayData
    setsym      :: Symbol # ∈ :hw101, i80
    trajdata    :: Trajdata
    frameind_lo :: Int # NOTE(tim): it takes a little while for cars to saturate the road
    frameind_hi :: Int #            these indeces tell us what that valid range is
end
abstract SceneGenerator

type ModelOptimizationResults{T<:SceneGenerator}
    params     :: Dict{Symbol, Any}
    logl_mean  :: Float64
    logl_stdev :: Float64
    score      :: Float64
    iter       :: Int
    CV_nfolds  :: Int
    CV_rounds  :: Int
end

const HIGHWAYDATA_SET = Dict{Symbol, HighwayData}()

function highwaydata_101a()
    if !haskey(HIGHWAYDATA_SET, :oneohonea)
        hd = HighwayData(:hw101, Trajdata(FILENAME_HW_101_0750), 476, 8797)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:oneohonea] = hd
    end
    HIGHWAYDATA_SET[:oneohonea]
end
function highwaydata_101b()
    if !haskey(HIGHWAYDATA_SET, :oneohoneb)
        hd = HighwayData(:hw101, Trajdata(FILENAME_HW_101_0805), 854, 9251)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:oneohoneb] = hd
    end
    HIGHWAYDATA_SET[:oneohoneb]
end
function highwaydata_101c()
    if !haskey(HIGHWAYDATA_SET, :oneohonec)
        hd = HighwayData(:hw101, Trajdata(FILENAME_HW_101_0820), 815, 9112)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:oneohonec] = hd
    end
    HIGHWAYDATA_SET[:oneohonec]
end
function highwaydata_80a()
    if !haskey(HIGHWAYDATA_SET, :eightya)
        hd = HighwayData(:i80, Trajdata(FILENAME_HW_80_0400), 993, 9063)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:eightya] = hd
    end
    HIGHWAYDATA_SET[:eightya]
end
function highwaydata_80b()
    if !haskey(HIGHWAYDATA_SET, :eightyb)
        hd = HighwayData(:i80, Trajdata(FILENAME_HW_80_0500), 681, 9458)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:eightyb] = hd
    end
    HIGHWAYDATA_SET[:eightyb]
end
function highwaydata_80c()
    if !haskey(HIGHWAYDATA_SET, :eightyc)
        hd = HighwayData(:i80, Trajdata(FILENAME_HW_80_0515), 1710, 9191)
        preprocess!(hd.trajdata)
        HIGHWAYDATA_SET[:eightyc] = hd
    end
    HIGHWAYDATA_SET[:eightyc]
end

const LANE_SPLINES = (Symbol=>Spline1D)[
        :hw101_1 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_1.jld", "spline"),
        :hw101_2 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_2.jld", "spline"),
        :hw101_3 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_3.jld", "spline"),
        :hw101_4 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_4.jld", "spline"),
        :hw101_5 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_5.jld", "spline"),
        :hw101_6 => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_101_6.jld", "spline"),
        :i80_1   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_1.jld",  "spline"),
        :i80_2   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_2.jld",  "spline"),
        :i80_3   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_3.jld",  "spline"),
        :i80_4   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_4.jld",  "spline"),
        :i80_5   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_5.jld",  "spline"),
        :i80_6   => load("/media/tim/DATAPART1/Data/NGSIM/output/splines/spline_80_6.jld",  "spline"),
    ]


show(io::IO, td::Trajdata) = @printf(io, "Trajdata")

carid_iter(trajdata::Trajdata) = keys(trajdata.car2start)
carid_set(trajdata::Trajdata)  = Set(keys(trajdata.car2start))
calc_ncars(trajdata::Trajdata) = length(trajdata.car2start)

carsinframe(td::Trajdata, frameind::Int) = td.frame2cars[frameind]
function car_df_index(td::Trajdata, carid::Int, frameind::Int)
    # NOTE(tim): given frameind and carind, find index of car in df
    #            Returns 0 if it does not exist

    df = td.df

    lo = td.car2start[carid]
    framestart = df[lo, :frameind]

    if framestart == frameind
        return lo
    elseif frameind < framestart
        return 0
    end

    n_frames = df[lo, :n_frames_in_dataset]
    target = frameind - framestart + lo
    if target > lo + n_frames
        return 0
    end

    target
end
iscarinframe(td::Trajdata, carid::Int, frameind::Int) = in(carid, carsinframe(td, frameind))

function get_lane_spline(sym::Symbol, laneid::Int)
    if sym == :hw101
        if laneid == 1
            return LANE_SPLINES[:hw101_1]
        elseif laneid == 2
            return LANE_SPLINES[:hw101_2]
        elseif laneid == 3
            return LANE_SPLINES[:hw101_3]
        elseif laneid == 4
            return LANE_SPLINES[:hw101_4]
        elseif laneid == 5
            return LANE_SPLINES[:hw101_5]
        elseif laneid == 6
            return LANE_SPLINES[:hw101_6]
        end
    elseif sym == :i80
        if laneid == 1
            return LANE_SPLINES[:i80_1]
        elseif laneid == 2
            return LANE_SPLINES[:i80_2]
        elseif laneid == 3
            return LANE_SPLINES[:i80_3]
        elseif laneid == 4
            return LANE_SPLINES[:i80_4]
        elseif laneid == 5
            return LANE_SPLINES[:i80_5]
        elseif laneid == 6
            return LANE_SPLINES[:i80_6]
        end
    end

    error("no such spline! $sym, $laneid")
end

get(td::Trajdata, sym::Symbol, dfind::Int) = td.df[dfind, sym]
get(td::Trajdata, sym::Symbol, carid::Int, frameind::Int) = get(td, sym, car_df_index(td, carid, frameind))
function get(td::Trajdata, carid::Int, frameind::Int, veh::Vehicle=Vehicle())
    dfind = car_df_index(td, carid, frameind)

    veh.x = td.df[dfind, :local_x]
    veh.y = td.df[dfind, :local_y]
    veh.v = td.df[dfind, :speed]
    veh.class  = td.df[dfind, :class ]
    veh.length = td.df[dfind, :length]
    veh.width  = td.df[dfind, :width ]

    veh.ϕ = 0.0
    if iscarinframe(td, carid, frameind+3)
        dfind2 = car_df_index(td, carid, frameind+3)
        x2 = td.df[dfind2, :local_x]
        y2 = td.df[dfind2, :local_y]
        veh.ϕ = atan2(x2-veh.x, y2-veh.y)
    elseif iscarinframe(td, carid, frameind-3)
        dfind2 = car_df_index(td, carid, frameind-3)
        x2 = td.df[dfind2, :local_x]
        y2 = td.df[dfind2, :local_y]
        veh.ϕ = atan2(veh.x-x2, veh.y-y2)
    end
    if abs(veh.ϕ) > 0.1
        veh.ϕ = randn()*0.01
    end

    veh
end
function get_info(td::Trajdata, carid::Int, frameind::Int, setsym::Symbol, info::VehicleExtraInfo=VehicleExtraInfo())
    
    @assert(setsym == :hw101 || setsym == :i80)

    dfind = car_df_index(td, carid, frameind)
    
    info.laneindex   = td.df[dfind, :lane]
    info.carind_fore = td.df[dfind, :carind_front]
    info.carind_rear = td.df[dfind, :carind_rear]

    laneindex = info.laneindex
    if 1 ≤ laneindex ≤ 6
        lanespline = get_lane_spline(setsym, info.laneindex)
        local_x = td.df[dfind, :local_x]
        local_y = td.df[dfind, :local_y]
        lane_x  = evaluate(lanespline, local_y)
        info.d_cl = local_x - lane_x
        info.fake_x = AVE_LANE_WIDTH * (laneindex-0.5) + info.d_cl
    else
        info.d_cl = Inf
        info.fake_x = 0.0
    end
    # info.d_cl = AVE_LANE_WIDTH * (laneindex-0.5) - td.df[dfind, :local_x]
    # info.fake_x = info.d_cl

    info
end

lanecenters(road::StraightRoadway) = ([1:road.nlanes]-0.5)*road.lanewidth
roadwidth(road::StraightRoadway) = road.nlanes*road.lanewidth
roadlength(road::StraightRoadway) = road.horizon_rear + road.horizon_fore

function get_lane_spline(td::Trajdata, laneid::Int;
    min_y :: Float64 = 0.0,
    max_y :: Float64 = 1000.0
    )

    count = 0
    for i = 1 : size(td.df, 1)
        if td.df[i, :lane] == laneid
            count += 1
        end
    end

    local_x_arr = Array(Float64, count)
    local_y_arr = Array(Float64, count)
    count = 0
    for i = 1 : size(td.df, 1)
        if td.df[i, :lane] == laneid
            count += 1
            local_x_arr[count] = td.df[i, :local_x]
            local_y_arr[count] = td.df[i, :local_y]
        end
    end

    disc = LinearDiscretizer(linspace(min_y, max_y, 1000))
    x_vars = Array(StreamStats.Var, length(y_values))
    for i = 1 : length(y_values)
        x_vars[i] = StreamStats.Var()
    end

    y_values = bincenters(disc)
    x_values = Array(Float64, length(y_values))
    for (x,y) in zip(local_x_arr, local_y_arr)
        bin = encode(disc, y)
        update!(x_vars[bin], x)
    end

    for i = 1 : length(y_values)
        x_values[i] = mean(var)
    end
    
    Spline1D(y_arr, x_arr, s=10.0)
end

function calc_vehicles_per_foot(td::Trajdata, lane::Int, frameind::Int)
    vehicles_per_foot = 0.0
    for carid in carsinframe(td, frameind)
        if get(td, :lane, carid, frameind) == lane
            vehicles_per_foot += 1
        end
    end
    vehicles_per_foot /= 2100
end
function calc_lanespeeds(scene::RoadScene)
    nlanes = scene.road.nlanes
    retval = zeros(Float64, nlanes)
    counts = zeros(Int, nlanes)

    for (i,veh) in enumerate(scene.vehicles)
        lane = scene.info[i].laneindex
        retval[lane] += veh.v
        counts[lane] += 1
    end

    retval ./ counts
end
function calc_lanedensities(scene::RoadScene)

    nlanes = scene.road.nlanes
    counts = zeros(Int, nlanes)

    for (i,info) in enumerate(scene.info)
        counts[info.laneindex] += 1
    end

    counts / roadlength(scene.road) # [vehicles / ft]
end
function calc_d_front(scene::RoadScene, carind::Int)
    @assert(scene.info[carind].carind_fore != 0)
    veh_front = scene.vehicles[scene.info[carind].carind_fore]
    veh_front.y - scene.vehicles[carind].y - veh_front.length
end
function calc_d_rear(scene::RoadScene, carind::Int)
    @assert(scene.info[carind].carind_rear != 0)
    veh_rear = scene.vehicles[scene.info[carind].carind_rear]
    veh = scene.vehicles[carind]
    veh.y - veh_rear.y - veh.length
end

function calc_laneindex(centers::Vector{Float64}, x::Float64)
    bestind = 0
    bestval = Inf
    for (i,center) in enumerate(centers)
        Δ = x - center
        if abs(Δ) < bestval
            bestval, bestind = abs(Δ), i
        end
    end
    bestind
end
function calc_vehicle_extra_info!(scene::RoadScene)

    nlanes = scene.road.nlanes
    centers = lanecenters(scene.road)

    vehicles = scene.vehicles
    nvehicles = length(vehicles)
    
    lanesets = Array(Set{Int}, nlanes)
    for i = 1 : nlanes
        lanesets[i] = Set{Int}()
    end

    sceneinfo = Array(VehicleExtraInfo, nvehicles)

    for (i,veh) in enumerate(vehicles)
        laneindex = calc_laneindex(centers, veh.x)
        d_cl      = veh.x - centers[laneindex]
        sceneinfo[i] = VehicleExtraInfo(laneindex, 0, d_cl, 0, 0)
        push!(lanesets[laneindex],i)
    end

    for i = 1 : nlanes
        carinds  = collect(lanesets[i])
        ncarinds = length(carinds)
        arr_y    = Array(Float64, ncarinds)
        for (j, carind) in enumerate(carinds)
            arr_y[j] = vehicles[carind].y
        end
        p = sortperm(arr_y)
        for laneorder = 1 : ncarinds
            carind = carinds[p[laneorder]]
            veh = vehicles[carind]
            info = sceneinfo[carind]
            info.laneorder = laneorder
            if laneorder > 1
                info.carind_rear = carinds[p[laneorder-1]]

                veh_rear = vehicles[sceneinfo[carind].carind_rear]
                info_rear = sceneinfo[sceneinfo[carind].carind_rear]
                # @assert(info_rear.carind_fore == carind)
                @assert(info_rear.laneorder == laneorder-1)
                @assert(veh_rear.y < veh.y)
                @assert(info_rear.laneindex == info.laneindex)
            end
            if laneorder < ncarinds
                info.carind_fore = carinds[p[laneorder+1]]

                veh_fore = vehicles[sceneinfo[carind].carind_fore]
                info_fore = sceneinfo[sceneinfo[carind].carind_fore]
                @assert(veh_fore.y > veh.y)
                @assert(info_fore.laneindex == info.laneindex)
            end
        end
    end

    scene.info = sceneinfo
    scene.lanespeeds = calc_lanespeeds(scene)
    scene.lanedensities = calc_lanedensities(scene)

    sceneinfo
end
function calc_sorted_cars_in_lane(scene::RoadScene, lane::Int)
    n_cars = 0
    for info in scene.info
        if info.laneindex == lane
            n_cars += 1
        end
    end

    if n_cars == 0
        return Int[]
    end

    retval = Array(Int, n_cars)
    for (i,info) in enumerate(scene.info)
        if info.laneindex == lane
            retval[info.laneorder] = i
        end
    end
    retval
end

function pull_101a(td::Trajdata, frameind::Int, 
    road::StraightRoadway = StraightRoadway(5, AVE_LANE_WIDTH, 200.0, 100.0))

    horizon_fore = road.horizon_fore
    horizon_rear = road.horizon_rear
    center_y = (DIST_HW_101_A - (horizon_fore + horizon_rear) )/2 + horizon_rear

    cars = carsinframe(td, frameind)
    vehicles = Array(Vehicle, length(cars))
    infoarr  = Array(VehicleExtraInfo, length(cars))
    count = 0
    for (i,carid) in enumerate(cars)

        lane = get(td, :lane, carid, frameind)
        if 1 ≤ lane ≤ 5
            veh = get(td, carid, frameind) 
            veh.y -= center_y
            if -horizon_rear < veh.y < horizon_fore

                @assert(veh.y + horizon_rear > 0.0)

                info = get_info(td, carid, frameind, :hw101)
                if abs(info.d_cl) < MAX_ABS_D_CL
                    count += 1
                    vehicles[count] = veh
                    infoarr[count] = info
                end
            end
        end
    end

    RoadScene(road, vehicles[1:count], infoarr[1:count])
end
function pull_101b(td::Trajdata, frameind::Int, 
    road::StraightRoadway = StraightRoadway(6, AVE_LANE_WIDTH, 200.0, 100.0))

    center_y = DIST_HW_101_A + DIST_HW_101_B / 2
    horizon_fore = road.horizon_fore
    horizon_rear = road.horizon_rear

    cars = carsinframe(td, frameind)
    vehicles = Array(Vehicle, length(cars))
    infoarr  = Array(VehicleExtraInfo, length(cars))
    count = 0
    for (i,carid) in enumerate(cars)

        lane = get(td, :lane, carid, frameind)
        if 1 ≤ lane ≤ 6
            veh = get(td, carid, frameind)
            veh.y -= center_y
            if -horizon_rear < veh.y < horizon_fore

                @assert(veh.y + horizon_rear > 0.0)

                info = get_info(td, carid, frameind, :hw101)
                if abs(info.d_cl) < MAX_ABS_D_CL
                    count += 1
                    vehicles[count] = veh
                    infoarr[count] = info
                end
            end
        end
    end

    RoadScene(road, vehicles[1:count], infoarr[1:count])
end
function pull_101c(td::Trajdata, frameind::Int,
    road::StraightRoadway = StraightRoadway(5, AVE_LANE_WIDTH, 200.0, 100.0))

    horizon_fore = road.horizon_fore
    horizon_rear = road.horizon_rear
    center_y = DIST_HW_101_A + DIST_HW_101_B + (DIST_HW_101_C - (horizon_fore + horizon_rear) )/2 + horizon_rear

    cars = carsinframe(td, frameind)
    vehicles = Array(Vehicle, length(cars))
    infoarr  = Array(VehicleExtraInfo, length(cars))
    count = 0
    for (i,carid) in enumerate(cars)

        lane = get(td, :lane, carid, frameind)
        if 1 ≤ lane ≤ 5
            veh = get(td, carid, frameind) 
            veh.y -= center_y
            if -horizon_rear < veh.y < horizon_fore

                @assert(veh.y + horizon_rear > 0.0)

                info = get_info(td, carid, frameind, :hw101)
                if abs(info.d_cl) < MAX_ABS_D_CL
                    count += 1
                    vehicles[count] = veh
                    infoarr[count] = info
                end
            end
        end
    end

    RoadScene(road, vehicles[1:count], infoarr[1:count])
end
function pull_80a(td::Trajdata, frameind::Int, 
    road::StraightRoadway = StraightRoadway(6, AVE_LANE_WIDTH, 200.0, 100.0))

    horizon_fore = road.horizon_fore
    horizon_rear = road.horizon_rear
    center_y = 100 + horizon_rear

    cars = carsinframe(td, frameind)
    vehicles = Array(Vehicle, length(cars))
    infoarr  = Array(VehicleExtraInfo, length(cars))
    count = 0
    for (i,carid) in enumerate(cars)

        lane = get(td, :lane, carid, frameind)
        if 1 ≤ lane ≤ road.nlanes
            veh = get(td, carid, frameind) 
            veh.y -= center_y
            if -horizon_rear < veh.y < horizon_fore

                @assert(veh.y + horizon_rear > 0.0)

                info = get_info(td, carid, frameind, :i80)
                if abs(info.d_cl) < MAX_ABS_D_CL
                    count += 1
                    vehicles[count] = veh
                    infoarr[count] = info
                end
            end
        end
    end

    RoadScene(road, vehicles[1:count], infoarr[1:count])
end
function pull_80b(td::Trajdata, frameind::Int,
    road::StraightRoadway = StraightRoadway(6, AVE_LANE_WIDTH, 200.0, 100.0))

    horizon_fore = road.horizon_fore
    horizon_rear = road.horizon_rear
    center_y = DIST_HW_80_A + DIST_HW_80_B - horizon_fore - 200

    cars = carsinframe(td, frameind)
    vehicles = Array(Vehicle, length(cars))
    infoarr  = Array(VehicleExtraInfo, length(cars))
    count = 0
    for (i,carid) in enumerate(cars)

        lane = get(td, :lane, carid, frameind)
        if 1 ≤ lane ≤ road.nlanes
            veh = get(td, carid, frameind) 
            veh.y -= center_y
            if -horizon_rear < veh.y < horizon_fore

                @assert(veh.y + horizon_rear > 0.0)

                info = get_info(td, carid, frameind, :i80)
                if abs(info.d_cl) < MAX_ABS_D_CL
                    count += 1
                    vehicles[count] = veh
                    infoarr[count] = info
                end
            end
        end
    end

    RoadScene(road, vehicles[1:count], infoarr[1:count])
end
function get_all_roadscenes(highwaydata::HighwayData, f_pull_roadscene::Function;
    frameskip::Int = 100
    )
    
    hi = highwaydata.frameind_hi
    lo = highwaydata.frameind_lo
    nframeinds = length(lo:frameskip:hi)
    scenes = Array(RoadScene, nframeinds)
    scenes[1] = f_pull_roadscene(highwaydata.trajdata, lo)
    road = scenes[1].road
    count = 1
    for i = lo+frameskip : frameskip : hi
        scenes[count+=1] = f_pull_roadscene(highwaydata.trajdata, i, road)
    end
    @assert(count == nframeinds)

    for scene in scenes
        calc_vehicle_extra_info!(scene)
    end

    scenes
end
function get_all_roadscenes(f_pull_roadscene::Function;
    frameskip :: Int = 100
    )

    if f_pull_roadscene == pull_101a || f_pull_roadscene == pull_101b ||
       f_pull_roadscene == pull_101c

        hdset = (highwaydata_101a(), highwaydata_101b(), highwaydata_101c())
    else
        hdset = (highwaydata_80a(), highwaydata_80b(), highwaydata_80c())
    end

    nscenes = 0
    for hd in hdset
        hi = hd.frameind_hi
        lo = hd.frameind_lo
        for _ in lo : frameskip : hi
            nscenes += 1
        end
    end
    scenes = Array(RoadScene, nscenes)

    scenes[1] = f_pull_roadscene(hdset[1].trajdata, hdset[1].frameind_lo)
    road = scenes[1].road

    count = 0
    for hd in hdset
        hi = hd.frameind_hi
        lo = hd.frameind_lo

        for i = lo : frameskip : hi
            scenes[count+=1] = f_pull_roadscene(hd.trajdata, i, road)
        end
    end

    for scene in scenes
        calc_vehicle_extra_info!(scene)
    end

    scenes
end

function get_arr_v(scenes::Vector{RoadScene})
    tot_nvehicles = 0
    for scene in scenes
        tot_nvehicles += length(scene.vehicles)
    end

    arr_v = Array(Float64, tot_nvehicles)

    count = 0
    for scene in scenes
        for (i,veh) in enumerate(scene.vehicles)
            count += 1
            arr_v[count] = veh.v
        end
    end

    arr_v
end
function get_arr_yaw(scenes::Vector{RoadScene})
    tot_nvehicles = 0
    for scene in scenes
        tot_nvehicles += length(scene.vehicles)
    end

    retval = Array(Float64, tot_nvehicles)

    count = 0
    for scene in scenes
        for (i,veh) in enumerate(scene.vehicles)
            count += 1
            retval[count] = veh.ϕ
        end
    end

    retval
end
function get_arr_d_cl(scenes::Vector{RoadScene})
    tot_nvehicles = 0
    for scene in scenes
        tot_nvehicles += length(scene.vehicles)
    end

    retval = Array(Float64, tot_nvehicles)

    count = 0
    for scene in scenes
        if isempty(scene.info)
            calc_vehicle_extra_info!(scene)
        end
        for (i,info) in enumerate(scene.info)
            count += 1
            info = sceneinfo[i]
            retval[count] = info.d_cl
        end
    end

    retval
end
function get_arr_d_front(scenes::Vector{RoadScene})

    tot_nvehicles = 0
    for scene in scenes
        tot_nvehicles += length(scene.vehicles)
    end

    retval = Array(Float64, tot_nvehicles)

    count = 0
    for scene in scenes
        if isempty(scene.info)
            calc_vehicle_extra_info!(scene)
        end
        for (i,info) in enumerate(scene.info)
            if info.carind_fore != 0
                retval[count+=1] =  calc_d_front(scene, i)
                @assert(retval[count] > -VEHICLE_MEAN_LENGTH)
            end
        end
    end

    retval[1:count]
end

function get_base_arrays(scenes::Vector{RoadScene})
    tot_nvehicles = 0
    for scene in scenes
        tot_nvehicles += length(scene.vehicles)
    end

    arr_v         = Array(Float64, tot_nvehicles)
    arr_d_front = Array(Float64, tot_nvehicles)
    arr_d_cl      = Array(Float64, tot_nvehicles)
    arr_yaw       = Array(Float64, tot_nvehicles)

    count = 0
    count_d_front = 0
    for scene in scenes
        for (i,veh) in enumerate(scene.vehicles)
            count += 1
            info = scene.info[i]
            arr_v[count]    = veh.v
            arr_d_cl[count] = info.d_cl
            arr_yaw[count]  = veh.ϕ
            if info.carind_fore != 0
                arr_d_front[count_d_front+=1] = calc_d_front(scene, i)
                # if !(arr_d_front[count_d_front] > 0.0)
                #     println("rear: ", veh)
                #     println("      ", info)
                #     println("fore: ", scene.vehicles[info.carind_fore])
                #     println("      ", scene.info[info.carind_fore])
                # end
                @assert(arr_d_front[count_d_front] > -scene.vehicles[info.carind_fore].length)
            end
        end
    end
    @assert(count == tot_nvehicles)

    # arr_v         += (rand(tot_nvehicles)-0.5)*0.02
    # arr_d_front += (rand(tot_nvehicles)-0.5)*0.02
    # arr_d_cl      += (rand(tot_nvehicles)-0.5)*0.02
    # arr_yaw       += (rand(tot_nvehicles)-0.5)*0.02

    (arr_v, arr_d_front[1:count_d_front], arr_d_cl, arr_yaw)
end
function get_scenes(sg::SceneGenerator, road :: StraightRoadway, nscenes::Int)
    retval = Array(RoadScene, nscenes)
    for i = 1 : nscenes
        retval[i] = generate_scene(sg, road)
    end
    retval
end

function loglikelihood(
    sg     :: SceneGenerator,
    scenes :: Vector{RoadScene}
    )

    retval = 0.0
    for scene in scenes
        retval += loglikelihood(sg, scene)
    end
    retval
end
function ave_crossvalidated_likelihood(scores::Matrix{Float64})
    arr = mean(scores, 1)
    μ = mean(arr)
    σ = stdm(arr, μ)
    (μ,σ)
end

include("trajdata_preprocess.jl")

include("univariate_scene_generator.jl")
include("joint_bn_simple_scene_generator.jl")
include("joint_bn_scene_generator.jl")
include("joint_bn_chain_scene_generator.jl")
include("joint_bn_chain_backwards_scene_generator.jl")
include("heirarchical_scene_generator.jl")
include("model_validation_metrics.jl")

include("incl_cairo_utils.jl")
include("trajdata_io.jl")

end # module