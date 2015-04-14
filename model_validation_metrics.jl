using StreamStats

const METRIC_DISCS = [
        :nvehicles                 => LinearDiscretizer(linspace(0, 60,31)),
        :noverlaps                 => LinearDiscretizer(linspace(0,  5, 6)),
        :ttc                       => LinearDiscretizer(linspace(0, 25,20)),
        :extreme_lanedensity_ratio => LinearDiscretizer(linspace(0,  5,20)),
        :extreme_lanespeed_ratio   => LinearDiscretizer(linspace(0,  5,20)),
    ]

function does_intersect_AABB(A::Vehicle, B::Vehicle)
    return (2*abs(A.x - B.x) < (A.width  + B.width)) &&
           (2*abs(A.y - B.y) < (A.length + B.length))
end

function compute_metrics(scenes::Vector{RoadScene})
    retval = Dict{Symbol,Any}()

    # n_overlaps
    #   if vehicle bounding box overlap
    #   not included in paper results because results were boring

    # ttc for each vehicle
    #   ttc = max(-d_front / (v_front - v_rear), 0.0)

    # nvehicles in blind spot
    #   true if vehicle 

    nscenes = length(scenes)
    retval[:nvehicles]   = Array(Float64, nscenes)
    retval[:noverlaps]   = Array(Float64, nscenes)
    retval[:ttc]         = Array(Float64, nscenes)
    retval[:extreme_lanedensity_ratio] = Array(Float64, nscenes)
    retval[:extreme_lanespeed_ratio]   = Array(Float64, nscenes)

    for (sceneindex,scene) in enumerate(scenes)

        nvehicles = length(scene.vehicles)

        retval[:nvehicles][sceneindex] = nvehicles

        var_ttc = StreamStats.Var()
        noverlaps = 0

        if isempty(scene.info)
            calc_vehicle_extra_info!(scene)
        end

        for (i,info) in enumerate(scene.info)
            veh = scene.vehicles[i]
            if info.carind_fore != 0
                d_front = calc_d_front(scene, i)
                if d_front ≤ 0.0
                    update!(var_ttc, 0.0)
                else
                    veh_front = scene.vehicles[info.carind_fore]
                    v_front = veh_front.v * cos(veh_front.ϕ)
                    
                    v_rear = veh.v * cos(veh.ϕ)
                    ttc = clamp(-d_front / (v_front - v_rear), 0.0, max(METRIC_DISCS[:ttc]))
                    update!(var_ttc, ttc)
                end
            end

            for j = i+1 : nvehicles
                veh2 = scene.vehicles[j]
                noverlaps += does_intersect_AABB(veh, veh2) ? 1 : 0
            end
        end

        retval[:noverlaps][sceneindex] = float64(noverlaps)
        retval[:ttc][sceneindex]       = mean(var_ttc)
        retval[:extreme_lanedensity_ratio][sceneindex] = scene.lanedensities[end]/scene.lanedensities[1]
        retval[:extreme_lanespeed_ratio][sceneindex]   = scene.lanespeeds[end]/scene.lanespeeds[1]
    end

    retval
end