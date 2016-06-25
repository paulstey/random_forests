# Random forests


""" 
Finds the threshold to split `features` with that minimizes the
mean-squared-error loss over `labels`.
Returns (best_val, best_thresh), where `best_val` is -MSE 

Variable name changes from original code 
  labels -> y 
  features -> x
  nl -> n_left 
  nr -> n_right
  su -> sum_y
  su2 -> sum_y2
  s_l -> sum_y_left
  s2_l -> sum_y2_left

"""
function _best_mse_loss{T<:Float64, U<:Real}(y::Vector{T}, x::Vector{U}, domain)
    # True, but costly assert. However, see
    # https://github.com/JuliaStats/StatsBase.jl/issues/164
    # @assert issorted(x) && issorted(domain) 
    best_val = -Inf
    best_thresh = 0.0
    n = length(y)

    
    sum_y_left = sum_y2_left = zero(T)            # scale values of 0
    
    sum_y = sum(y)::T             # scalar sum of all y_i
    sum_y2 = zero(T);                  # scalar value of 0
    
    # sum of squares
    for i = 1:n 
        sum_y2 += y[i] * y[i] 
    end  

    n_left = 0
    i = 1
    # Because the `x` are sorted, below is an O(N) algorithm for finding
    # the optimal threshold amongst `domain`. We simply iterate through the
    # array and update sum_y_left and s_r (= sum(y) - sum_y_left) as we go. - @cstjean
    @inbounds for thresh in domain
        
        # this loop checks which side of the split this x_i is on
        while i <= n && x[i] < thresh

            sum_y_left += y[i]
            sum_y2_left += y[i]^2
            n_left += 1

            i += 1
        end
        s_r = sum_y - sum_y_left
        s2_r = sum_y2 - sum_y2_left
        
        n_right = n - n_left

        # This check is necessary I think because in theory all y could
        # be the same, then either n_left or n_right would be 0. - @cstjean
        if n_right > 0 && n_left > 0
            loss = sum_y2_left - sum_y_left^2/n_left + s2_r - s_r^2/n_right
            if -loss > best_val
                best_val = -loss
                best_thresh = thresh
            end
        end
    end
    return best_val, best_thresh
end

