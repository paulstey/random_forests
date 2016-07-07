
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

# n = 1000
# y = rand([true, false], n);
# @time majority_vote(y)

"""
This was formerly called neg_z1_loss() in Ben's original DecisionTree code,
and the variable `y` was formerlly called `labels`
"""
function _classifcation_error_loss(y::Vector, weights::Vector)
    missmatches = y .!= majority_vote(y)
    loss = sum(weights[missmatches])
    return -loss
end

# This method is dispatched when weights are omitted. This
# allows us to compute the loss function 5x faster
function _classifcation_error_loss(y::Vector)
    loss = sum(y)
    return -loss
end


n = 1000
y = rand([true, false], n);
wgt = repeat([1], inner = n);

@time _classifcation_error_loss(y)
