
const SMOOTHING_WIDTH_ACCEL = 4.0 # [s]
const SMOOTHING_WIDTH_SPEED = 1.0 # [s]
const SMOOTHING_WIDTH_POS   = 0.5 # [s]

# from "Estimating Acceleration and Lane-Changing
#       Dynamics Based on NGSIM Trajectory Data"
function symmetric_exponential_moving_average(
    arr :: Vector{Float64},
    T   :: Float64; # smoothing width [s]
    dt  :: Float64 = TIMESTEP # sampling period [s]
    )

    Δ = T / dt

    N = length(arr)
    retval = Array(Float64, N)

    for i = 1 : N

        Z = 0.0
        x = 0.0

        D = min(int(3Δ), i-1)
        if i+D > N
            D = N-i
        end

        for k in (i-D):(i+D)
            e = exp(-abs(i-k)/Δ)
            Z += e
            x += arr[k] * e
        end

        retval[i] = x / Z
    end

    retval
end
function preprocess!(td::Trajdata)

    for carid in carid_set(td)

        dfstart = td.car2start[carid]
        N = td.df[dfstart, :n_frames_in_dataset]

        x_arr = Array(Float64, N)
        y_arr = Array(Float64, N)
        v_arr = Array(Float64, N)

        for i = 1 : N
            x_arr[i] = td.df[dfstart + i - 1, :local_x]
            y_arr[i] = td.df[dfstart + i - 1, :local_y]
            v_arr[i] = td.df[dfstart + i - 1, :speed]
        end

        x_arr2 = symmetric_exponential_moving_average(x_arr, SMOOTHING_WIDTH_POS)
        y_arr2 = symmetric_exponential_moving_average(y_arr, SMOOTHING_WIDTH_POS)
        v_arr2 = symmetric_exponential_moving_average(v_arr, SMOOTHING_WIDTH_SPEED)

        for i = 1 : N
            td.df[dfstart + i - 1, :local_x] = x_arr2[i]
            td.df[dfstart + i - 1, :local_y] = y_arr2[i]
            td.df[dfstart + i - 1, :speed]   = v_arr2[i]
        end        
    end
end