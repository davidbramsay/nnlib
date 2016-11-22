#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef struct layer{

    int size;
    matrix_t *W;
    matrix_t *thresh;

} layer_t;



typedef struct net{

    layer_t *layers;
    int num_layers;

    double momentum;
    double learning_rate;
    double cooling_rate; //divisor of learning_rate each training cycle

} net_t;



typedef struct layerout{

    int size;
    matrix_t *output; //output of neurons in layer, 1xSIZE
    matrix_t *error; //error in output given a target backproped through, 1xSIZE
    matrix_t *d_W; //previous weight matrix delta, PREVLAYER_SIZExSIZE
    matrix_t *cum_d_W; //cumulative delta updates, PREVLAYER_SIZExSIZE

} layerout_t;



typedef struct trainingvals{

    int num_layers;
    layerout_t *layerouts;

} trainingvals_t;


// net_t *n;

//*n
// = {layers = 0x100102920, num_layers = 4, momentum = 0.5,
//                  learning_rate = 0.5, cooling_rate = 0.5}

//n->layers[1]
// = {size = 10, W = 0x100300270, thresh = 0x100300250}

//n->layers[1].W  //similar for thresh, but thresh is always num_rows=1 and addressed [0][n]
// = (matrix_t *) 0x100300270

//*(n->layers[1].W)
// = {vals = 0x100102ae0, num_rows = 5, num_cols = 10}

//n->layers[1].W->vals[4][8]
// = 0

//n->layers[0].thresh->vals[0][2]
// = 0


#endif /* NN_H */
