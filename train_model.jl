@everywhere using Trajdata_
using HDF5, JLD

reload("utils.jl")

const OUTPUT_FOLDER_JLD = "/media/tim/DATAPART1/Data/NGSIM/output/models/"
const OUTPUT_SECTIONBARPLOT = "/home/tim/Documents/wheelerworkspace/Papers/2015_TrafficSceneDistributions/data-set-sectionbarplot.csv"
const OUTPUT_101B_BASE = "/home/tim/Documents/wheelerworkspace/Papers/2015_TrafficSceneDistributions/data-set-"

const CV_nfolds = 10
const CV_rounds = 10
const N_SCENES_TO_GENERATE = 1000
const N_SCENES_TO_OUTPUT   = 5

const PERCENT_TRAIN = 0.9
const CONFIDENCE = 2.228 # 95% (for 10 samples)

result_dict = Dict{String, Trajdata_.ModelOptimizationResults}()

df_sectionbarplot = readtable(OUTPUT_SECTIONBARPLOT)

for (setname, f_pull) in (
    ("101b", Trajdata_.pull_101b),
    ("101a", Trajdata_.pull_101a),
    ("101c", Trajdata_.pull_101c),
    ( "80a", Trajdata_.pull_80a),
    ( "80b", Trajdata_.pull_80b),
    )

    scenes = Trajdata_.get_all_roadscenes(f_pull)
    nscenes = length(scenes)
    println("nscenes $setname: ", nscenes)

    p = randperm(nscenes)
    splitind = int(nscenes * PERCENT_TRAIN)
    train = scenes[p[1:splitind]]
    validate = scenes[p[splitind+1:end]]

    for T in (
        Trajdata_.HeirarchicalSceneGenerator,
        Trajdata_.JointBNChainSceneGenerator,
        Trajdata_.JointBNSimpleSceneGenerator,
        Trajdata_.UnivariateSceneGenerator, 
        )

        modelname = Trajdata_.model_name(T)
        println("MODEL: ", modelname)

        tic()
        res = Trajdata_.cyclic_coordinate_ascent_parallel(T, scenes,
                    CV_rounds = CV_rounds, CV_nfolds =CV_nfolds)
        toc()
        result_dict[setname] = res
        output_file = @sprintf("%smodeloptres_%s_%s.jld", OUTPUT_FOLDER_JLD, modelname, setname)
        JLD.save(output_file, "res", res)
        Trajdata_.write(res, setname)

        # update df_sectionbarplot
        ind_set = findfirst(df_sectionbarplot[:sets], setname)
        println(ind_set)
        df_sectionbarplot[ind_set, symbol(modelname)] = res.logl_mean
        df_sectionbarplot[ind_set, symbol(modelname*"error")] = res.logl_stdev * CONFIDENCE
        writetable_noquotemark(OUTPUT_SECTIONBARPLOT, df_sectionbarplot)

        # train a model on all scenes
        sg = Trajdata_.train(res, scenes)
        sample_scenes = Trajdata_.get_scenes(sg, scenes[1].road, N_SCENES_TO_GENERATE)
        metrics = Trajdata_.compute_metrics(sample_scenes)
        Trajdata_.write(sg, metrics, setname)
        Trajdata_.output_scenes(sg, sample_scenes[1:N_SCENES_TO_OUTPUT], setname)

        output_file2 = @sprintf("%smodeloptstruct_%s_%s.txt", OUTPUT_FOLDER_JLD, modelname, setname)
        open(output_file2, "w") do fout
            if T == Trajdata_.UnivariateSceneGenerator
                println("d_front: ", nlabels(sg.d_front.disc))
                println("v: ",       nlabels(sg.v.disc))
                println("d_cl: ",    nlabels(sg.d_cl.disc))
                println("yaw:  ",    nlabels(sg.yaw.disc))
            elseif T == Trajdata_.HeirarchicalSceneGenerator || T == Trajdata_.JointBNChainSceneGenerator || T == Trajdata_.JointBNSimpleSceneGenerator
                for (sym, i) in sg.varindeces
                    println(fout, sym, "{",  Discretizers.nlabels(sg.discmap[sym]), "}:")
                    for j in Smile.get_children(sg.net, i )
                        sym2 = :none
                        for (s2, i2) in sg.varindeces
                            if i2 == j
                                sym2 = s2
                            end
                        end
                        println(fout, "\t", sym2, "{",  Discretizers.nlabels(sg.discmap[sym2]), "}")
                    end
                end
            end
        end

        if setname == "101b"
            centers, counts = Trajdata_.get_histogram_line(:nvehicles, metrics[:nvehicles])
            writetable_noquotemark(OUTPUT_101B_BASE*"nvehicles-"*modelname*".csv", DataFrame(x=centers, y=counts))
            centers, counts = Trajdata_.get_histogram_line(:noverlaps, metrics[:noverlaps])
            writetable_noquotemark(OUTPUT_101B_BASE*"noverlaps-"*modelname*".csv", DataFrame(x=centers, y=counts))
            centers, counts = Trajdata_.get_histogram_line(:ttc, metrics[:ttc])
            writetable_noquotemark(OUTPUT_101B_BASE*"ttc-"*modelname*".csv", DataFrame(x=centers, y=counts))
            centers, counts = Trajdata_.get_histogram_line(:extreme_lanespeed_ratio, metrics[:extreme_lanespeed_ratio])
            writetable_noquotemark(OUTPUT_101B_BASE*"extremelanespeedratio-"*modelname*".csv", DataFrame(x=centers, y=counts))
            centers, counts = Trajdata_.get_histogram_line(:extreme_lanedensity_ratio, metrics[:extreme_lanedensity_ratio])
            writetable_noquotemark(OUTPUT_101B_BASE*"extremelanedensityratio-"*modelname*".csv", DataFrame(x=centers, y=counts))
        end
    end
end

using StreamStats
modelname = "realworld"
for (setname, f_pull) in (
    ("101a", Trajdata_.pull_101a),
    ("101b", Trajdata_.pull_101b),
    ("101c", Trajdata_.pull_101c),
    ( "80a", Trajdata_.pull_80a),
    ( "80b", Trajdata_.pull_80b),
    )

    scenes = Trajdata_.get_all_roadscenes(f_pull)
    metrics = Trajdata_.compute_metrics(scenes)
    # Trajdata_.write(metrics, setname)

    
    ld = StreamStats.Var() # mean lane density
    ls = StreamStats.Var() # mean lane speed
    for s in scenes
        for d in s.lanedensities
            if !isnan(d) && !isinf(d)
                update!(ld, d)
            end
        end
        for v in s.lanespeeds
            if !isnan(v) && !isinf(v)
                update!(ls, v)
            end
        end
    end
    println(setname)
    println("mean lane density (vehicles / ft): ", mean(ld), " +- ", sqrt(ld.v_hat))
    println("                  (vpm)          : ", mean(ld)/0.000189394, " +- ", sqrt(ld.v_hat)/0.000189394)
    println("mean lane speed   (ft/s):          ", mean(ls), " +- ", sqrt(ls.v_hat))

    p = randperm(length(scenes))
    Trajdata_.output_scenes(scenes[p[1:N_SCENES_TO_OUTPUT]], setname)

    if setname == "101b"
        centers, counts = Trajdata_.get_histogram_line(:nvehicles, metrics[:nvehicles])
        writetable_noquotemark(OUTPUT_101B_BASE*"nvehicles-"*modelname*".csv", DataFrame(x=centers, y=counts))
        centers, counts = Trajdata_.get_histogram_line(:noverlaps, metrics[:noverlaps])
        writetable_noquotemark(OUTPUT_101B_BASE*"noverlaps-"*modelname*".csv", DataFrame(x=centers, y=counts))
        centers, counts = Trajdata_.get_histogram_line(:ttc, metrics[:ttc])
        writetable_noquotemark(OUTPUT_101B_BASE*"ttc-"*modelname*".csv", DataFrame(x=centers, y=counts))
        centers, counts = Trajdata_.get_histogram_line(:extreme_lanespeed_ratio, metrics[:extreme_lanespeed_ratio])
        writetable_noquotemark(OUTPUT_101B_BASE*"extremelanespeedratio-"*modelname*".csv", DataFrame(x=centers, y=counts))
        centers, counts = Trajdata_.get_histogram_line(:extreme_lanedensity_ratio, metrics[:extreme_lanedensity_ratio])
        writetable_noquotemark(OUTPUT_101B_BASE*"extremelanedensityratio-"*modelname*".csv", DataFrame(x=centers, y=counts))
    end
end
