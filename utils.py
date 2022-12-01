import numpy as np

def projection_zero_min_bounds(vector, volumes, total_volume) :
    u = vector.clip(min=0.)
    if(np.sum(u*volumes)<=total_volume):
        return u
    
    sorted_indexes = np.argsort(vector/volumes)
    
    a = volumes[sorted_indexes[-1]]**2
    b = (volumes*vector)[sorted_indexes[-1]]
    for i in range(len(u)-2,-1,-1) :
        if( (b-total_volume)/a > (vector/volumes)[sorted_indexes[i]]) :
            break
        a += volumes[sorted_indexes[i]]**2
        b += (volumes*vector)[sorted_indexes[i]]

    theta = (b-total_volume)/a
    output = (vector-theta*volumes).clip(min=0.)

    #print("Theta: {}, Slackness condition: {}, Theta constraint: {}".format(theta, theta*(np.sum(volumes*output)-total_volume), np.sum(volumes*output)-total_volume))
    return output

def projection(vector, volumes, total_volume, min_bounds=0.) :
    new_total_volume = total_volume-np.sum(volumes*min_bounds)
    assert new_total_volume>=0., "Bounds do not respect the constraint: np.sum(volumes*min_bounds) <= total_volume"
    return projection_zero_min_bounds(vector-min_bounds, volumes, new_total_volume)+min_bounds