# Random forests
using RDatasets

""" 
Finds the threshold to split `features` with that minimizes the
mean-squared-error loss over `labels`.
Returns (best_val, best_thresh), where `best_val` is -MSE 

Variable name changes from original code 
  labels   -> y 
  features -> x
  nl       -> n_left 
  nr       -> n_right
  su       -> sum_y
  su2      -> sum_y2
  s_l      -> sum_y_left
  s2_l     -> sum_y2_left
  s_r      -> sum_y_right 
  s2_r     -> sum_y2_right

"""
function _best_mse_loss{T<:Float64, U<:Real}(y::Vector{T}, x::Vector{U}, domain)
    best_val = -Inf
    best_thresh = 0.0
    n = length(y)

    sum_y_left = sum_y2_left = zero(T)            # scale values of 0
    
    sum_y = sum(y)::T                             # scalar sum of all y_i
    sum_y2 = zero(T);                             # scalar value of 0
    
    # get sum of squares
    for i = 1:n 
        sum_y2 += y[i]^2 
    end  

    n_left = 0
    i = 1

    # Since`x` is sorted, below is an O(n) algorithm for finding the optimal 
    # threshold in `domain`. We iterate through the array and update sum_y_left 
    # and sum_y_right (= sum(y) - sum_y_left) as we go. - @cstjean
    @inbounds @simd for thresh in domain
        
        # this loop checks which side of the split this x_i is on
        while i <= n && x[i] < thresh
            sum_y_left += y[i]
            sum_y2_left += y[i]^2
            n_left += 1
            i += 1
        end
        sum_y_right = sum_y - sum_y_left
        sum_y2_right = sum_y2 - sum_y2_left
        n_right = n - n_left

        # This check is necessary I think because in theory all y could
        # be the same, then either n_left or n_right would be 0. - @cstjean
        if n_right > 0 && n_left > 0
            
            # This isn't really squared-error loss. We're computing the sum of variance 
            # in the daughter nodes times the N for that daughter node. This last point 
            # is why the two paranthetical terms here look a bit diffent than mere variance. 
            # We have canceled a constant term N for each term.
            loss = (sum_y2_left - sum_y_left^2/n_left) + (sum_y2_right - sum_y_right^2/n_right)
            
            # update best value and threshold
            if -loss > best_val
                best_val = -loss
                best_thresh = thresh
            end
        end
    end
    return best_val, best_thresh
end


"""
Variable name changes from original code 
  labels     -> y 
  features   -> x
  nr         -> n 
  nf         -> p 
  i          -> j
  features_i -> x_j
  labels_i   -> y_ord
  domain_i   -> domain_j

"""
function _split_mse{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}, nsubfeatures::Int)
    n, p = size(features)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(p)
        inds = r[1:nsubfeatures]
    else
        inds = 1:p
    end

    for j in inds
        # Sorting used to be performed only when n <= 100, but doing it
        # unconditionally improved fitting performance by 20%. It's a bit of a
        # puzzle. Either it improved type-stability, or perhaps branch
        # prediction is much better on a sorted sequence.
        ord = sortperm(features[:, j])
        x_j = x[ord, j]
        y_ord = y[ord]
        

        if n > 100
            if VERSION >= v"0.4.0-dev"
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99); sorted=true)
            else  # sorted=true isn't supported on StatsBase's Julia 0.3 version
                domain_j = quantile(x_j, linspace(0.01, 0.99, 99))
            end
        else
            domain_j = x_j
        end
        
        value, thresh = _best_mse_loss(y_ord, x_j, domain_j)
        
        if value > best_val
            best_val = value
            best = (j, thresh)
        end
    end

    return best
end







d = dataset("datasets", "airquality")
dc = d[complete_cases(d), :]
features = convert(Array, dc[:, 2:6]);
labels = convert(Array, dc[:, 1]);

