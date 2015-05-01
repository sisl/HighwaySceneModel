using DataFrames

function printtable_noquotemark(
    io::IO,
    df::AbstractDataFrame;
    header::Bool = true,
    separator::Char = ',',
    )

    n, p = size(df)
    etypes = eltypes(df)
    if header
        cnames = DataFrames._names(df)
        for j in 1:p
            print(io, cnames[j])
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    for i in 1:n
        for j in 1:p
            if ! (etypes[j] <: Real)
                DataFrames.escapedprint(io, df[i, j], "\"'")
            else
                print(io, df[i, j])
            end
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    return
end
function printtable_noquotemark(df::AbstractDataFrame;
    header::Bool = true,
    separator::Char = ',',
    )

    printtable_noquotemark(STDOUT,
               df,
               header = header,
               separator = separator
               )
    return
end
function writetable_noquotemark(filename::String,
    df::AbstractDataFrame;
    header::Bool = true,
    separator::Char = DataFrames.getseparator(filename),
    append::Bool = false)

    if endswith(filename, ".bz") || endswith(filename, ".bz2")
        throw(ArgumentError("BZip2 compression not yet implemented"))
    end

    if append && isfile(filename) && filesize(filename) > 0
        file_df = readtable(filename, header = false, nrows = 1)

        # Check if number of columns matches
        if size(file_df, 2) != size(df, 2)
            throw(DimensionMismatch("Number of columns differ between file and DataFrame"))
        end

        # When 'append'-ing to a nonempty file,
        # 'header' triggers a check for matching colnames
        if header
            if any(i -> symbol(file_df[1, i]) != index(df)[i], 1:size(df, 2))
                throw(KeyError("Column names don't match names in file"))
            end

            header = false
        end
    end

    openfunc = endswith(filename, ".gz") ? gzopen : open

    openfunc(filename, append ? "a" : "w") do io
        printtable_noquotemark(io,
                   df,
                   header = header,
                   separator = separator
                   )
    end

    return
end

# opens a csv file as a DataFrame, replaces a particular column with given data,
# and re-writes the file
#  DOES NOT CHECK THAT THE TYPES MATCH
#  WILL NOT DO ANYTHING IF THE COLUMN LENGTH IS INCORRECT and will give a warning
# returns whether or not it succeeded (true on success)
function rewrite_column_on_disk(csvfilename::String, column::Symbol, columndata::Vector)

    if csvfilename[end-3:end] != ".csv" || !isfile(csvfilename) || !isreadable(csvfilename) || !iswritable(csvfilename)
        return false
    end

    df = readtable(csvfilename)
    if !haskey(df, column)
        return false
    elseif size(df, 1) != length(columndata)
        return false
    end

    df[column] = columndata
    writetable_noquotemark(csvfilename, df)
    true
end
function rewrite_value_on_disk(csvfilename::String, column::Symbol, index::Int, value::Any)

    if index ≤ 0
        return false
    end

    if csvfilename[end-3:end] != ".csv" || !isfile(csvfilename) || !isreadable(csvfilename) || !iswritable(csvfilename)
        return false
    end

    df = readtable(csvfilename)
    if !haskey(df, column)
        return false
    elseif index > size(df, 1)
        return false
    end

    df[index,column] = value
    writetable_noquotemark(csvfilename, df)
    true
end
