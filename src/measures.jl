



function _hist_add!{T}(counts::Dict{T,Int}, y::Vector{T}, region::UnitRange{Int})
    for i in region
        lbl = y[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end



"""
Given a vector of labels, this returns a Dict where the keys are
the unique y[i] and the values are the number of times that y[i]
occurs in the vector y.
"""
_hist{T}(y::Vector{T}, region::UnitRange{Int} = 1:endof(y)) = _hist_add!(Dict{T,Int}(), y, region)


"""
Given the vector y, this simply returns the value that 
appears most frequently in y.
"""
function majority_vote(y::Vector)
    if length(y) == 0
        return nothing
    end
    counts = _hist(y)
    top_vote = y[1]
    top_count = -1
    for (k,v) in counts
        if v > top_count
            top_vote = k
            top_count = v
        end
    end
    return top_vote
end
