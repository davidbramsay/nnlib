#include "matrix.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//TODO:
//(1) UPDATE THRESHOLDS
//(2) ALLOW YOU TO DEFINE WHAT TYPE OF FUNCTION EACH LAYER HAS
//(3) PREPROCESSING/NORMALIZATION OF INPUT DATA
//(4) SUPPORT BINARY FILE INPUT, NOT JUST TXT

//first layer is totally passive, just there to make sure it matches the size
//of the input vector.  It's Weight and Thresh matrices are set to NULL, no
//function is applied to this first step.

//#define DEBUG



double sigmoid(double val){
//calc sigmoid given a value

    return 1.0 / (1.0 + exp(-val));
}



double d_sigmoid(double val){
//calc derivative of sigmoid

    //since we're calling this on the output of the neurons, val has already
    //been passed through a sigmoid function
    return val * (1 - val);
}



//can also use tanh, built-in function, instead of sigmoid
double d_tanh(double val){
//calc derivative of tanh

    //since we're calling this on the output of the nerons, val has already
    //been passed through a tanh function
    return 1.0 - (val*val);
}



void m_randomize_weights(matrix_t *matrix, int upper, int lower){
//fill matrix will random values between lower and upper

    srand(time(NULL));

    for (int i = 0; i < matrix->num_rows; i++)
        for (int j = 0; j < matrix->num_cols; j++)
            matrix->vals[i][j] = ((double)(upper-lower) * (double)rand() / RAND_MAX) + lower;
}



void nn_randomize_weights(net_t *net, int upper, int lower){
//fill all weight matrices in a nn object with random values between lower and
//upper

    for (int i = 1; i < net->num_layers; i++){
        m_randomize_weights(net->layers[i].W, upper, lower);
    }
}



net_t *nn_init(int num_layers, int layer_sizes[], double momentum, double learning_rate, double cooling_rate){
//initialize a neural net object

    net_t *nn = malloc(sizeof(net_t));

    nn->momentum = momentum;
    nn->learning_rate = learning_rate;
    nn->cooling_rate = cooling_rate;

    nn->num_layers = num_layers;
    nn->layers = malloc(num_layers * sizeof(layer_t));

    for(int i = 0; i < num_layers; i++){

        nn->layers[i].size = layer_sizes[i];

        if (i) {
            nn->layers[i].W = m_init(layer_sizes[i-1], layer_sizes[i]);
            nn->layers[i].thresh = m_init(1, layer_sizes[i]);
        } else { //input layer
            nn->layers[i].W = NULL;
            nn->layers[i].thresh = NULL;
        }

    }

    return nn;
}



trainingvals_t *nn_init_trainingvals(net_t* net){
//initialize a training vals obj (used in forward/back prop) for a given net

    trainingvals_t *result = malloc(sizeof(trainingvals_t));

    result->num_layers = net->num_layers;
    result->layerouts = malloc(result->num_layers * sizeof(layerout_t));

    for(int i = 0; i < result->num_layers; i++){

        result->layerouts[i].size = net->layers[i].size;

        result->layerouts[i].output = NULL;
        result->layerouts[i].error = NULL;

        if (i) {
            result->layerouts[i].d_W = m_init(net->layers[i].W->num_rows, net->layers[i].W->num_cols);
            result->layerouts[i].cum_d_W = m_init(net->layers[i].W->num_rows, net->layers[i].W->num_cols);
        } else { //input layer
            result->layerouts[i].d_W = NULL;
            result->layerouts[i].cum_d_W = NULL;
        }

    }

    return result;
}



void nn_trainingvals_free(trainingvals_t *vals){
//free memory associated with a trainingvals struct

    for (int i = 0; i < vals->num_layers; i++){

        if (vals->layerouts[i].output)
            m_free(vals->layerouts[i].output);

        if (vals->layerouts[i].error)
            m_free(vals->layerouts[i].error);

        if (vals->layerouts[i].cum_d_W)
            m_free(vals->layerouts[i].cum_d_W);

        if (vals->layerouts[i].d_W)
            m_free(vals->layerouts[i].d_W);
    }

    free(vals->layerouts);
    free(vals);

}



void nn_print(net_t *net){
//pretty print of nn object

    printf("------------------------------------------------------------------------\n");
    printf("network has %d layers. learning: %.2f, momentum: %.2f, cooling: %.2f.\n\n",
            net->num_layers, net->learning_rate, net->momentum, net->cooling_rate);

    printf("input layer: %d neurons ->", net->layers[0].size);
    for (int i = 1; i < net->num_layers - 1; i++)
        printf(" hidden layer #%d: %d neurons ->", i, net->layers[i].size);
    printf(" output layer: %d neurons\n", net->layers[net->num_layers - 1].size);

    printf("------------------------------------------------------------------------\n");

    for (int i = 1; i < net->num_layers; i++){
        printf(">>> weights connect layer %d (size %d) to layer %d (size %d)\n",
            i-1, net->layers[i-1].size, i, net->layers[i].size);
        m_print(net->layers[i].W);

        printf(">>> layer %d threshold values:\n", i);
        m_print(net->layers[i].thresh);
    }

    printf("------------------------------------------------------------------------\n\n");
}



double nn_train(net_t *net, matrix_t *input, matrix_t *goal){
//actual training and backprop step for each batch
//returns max error

    static int batch = 0;
    static double learning_rate = 0;
    static trainingvals_t *tvals = NULL;

    if (learning_rate ==0) learning_rate = net->learning_rate;
    if (tvals == NULL) tvals = nn_init_trainingvals(net);

    double batch_max_error = 0;
    matrix_t *error, *target_output, *delta_W = NULL;

    for (int i = 0; i < input->num_rows; i++) {

        tvals->layerouts[0].output = m_getrow(input, i);
        target_output = m_getrow(goal, i);

        //forward pass
        for (int j = 1; j < net->num_layers; j++) {

            #ifdef DEBUG
            printf("%d:fw pass #1 - prev activation\n", i);
            m_print(tvals->layerouts[j-1].output);
            printf("%d:fw pass #2 - thresh\n", i);
            m_print(net->layers[j].thresh);
            printf("%d:fw pass #3 - prev matrix\n", i);
            m_print(net->layers[j].W);
            #endif

            tvals->layerouts[j].output = m_multi(tvals->layerouts[j-1].output, net->layers[j].W);
            m_padd(tvals->layerouts[j].output, net->layers[j].thresh);

            #ifdef DEBUG
            printf("%d:fw pass #4 - matrix after multiply and add thresh\n", i);
            m_print(tvals->layerouts[j].output);
            #endif

            m_apply(tvals->layerouts[j].output, &sigmoid);

            #ifdef DEBUG
            printf("%d:fw pass #5 - after apply sigmoid\n", i);
            m_print(tvals->layerouts[j].output);
            #endif

        }

        #ifdef DEBUG
        printf("Finished forward pass #%d...\n", i);
        #endif

        //backprop
        for (int j = (net->num_layers - 1); j > 0; j--) {

            //move derivative of current func/activation into error matrix
            tvals->layerouts[j].error = m_copy(tvals->layerouts[j].output); //copy layer output to error matrix
            m_apply(tvals->layerouts[j].error, &d_sigmoid); //p-derivative of output through activation func

            #ifdef DEBUG
            printf("%d:bp pass #1 - output from layer\n", i);
            m_print(tvals->layerouts[j].output);
            printf("%d:bp pass #2 - apply derivative of sigmoid and copy to error\n", i);
            m_print(tvals->layerouts[j].error);
            #endif

            //multiply p-derivative of layer func with prev layer error
            if (j == (net->num_layers - 1)) { //output layer

                #ifdef DEBUG
                printf("%d:bp pass #3 (for output layer) - output\n", i);
                m_print(tvals->layerouts[j].output);
                printf("%d:bp pass #4 (for output layer) - target\n", i);
                m_print(target_output);
                #endif

                //calc final error, (output - target)
                error = m_copy(tvals->layerouts[j].output);
                m_psub(error, target_output);
                m_free(target_output);

                if (fabs(m_dmaxval(error)) > batch_max_error)
                    batch_max_error = fabs(m_dmaxval(error));

                #ifdef DEBUG
                printf("%d:bp pass #5 (for output layer) - error diff\n", i);
                m_print(error);
                printf("%d:bp pass #5 (for output layer) - max error: %f\n", i, batch_max_error);
                #endif

            } else { //hidden layer

                #ifdef DEBUG
                printf("%d:bp pass #3 - prev error\n", i);
                m_print(tvals->layerouts[j+1].error);
                printf("%d:bp pass #4 - weights\n", i);
                m_print(net->layers[j+1].W);
                #endif

                //calc error, weights * prev layer error
                matrix_t *W_t = m_transpose(net->layers[j + 1].W);
                error = m_multi(tvals->layerouts[j + 1].error, W_t);
                m_free(W_t);

                #ifdef DEBUG
                printf("%d:bp pass #5 - error diff\n", i);
                m_print(error);
                #endif

            }

            m_pmulti(tvals->layerouts[j].error, error); //p-derivative point multiply by error
            m_free(error);

            #ifdef DEBUG
            printf("%d:bp pass #6 - final error for layer, error.*d_sigmoid\n", i);
            m_print(tvals->layerouts[j].error);
            #endif
        }

        #ifdef DEBUG
        printf("Finished backprop pass #%d...\n", i);
        #endif

        //accumulate delta W values
        for (int j = 1; j < net->num_layers; j++) {

            matrix_t *activation_t = m_transpose(tvals->layerouts[j-1].output); 
            delta_W = m_multi(activation_t, tvals->layerouts[j].error);
            m_free(activation_t);

            #ifdef DEBUG
            printf("%d:d_W #1 - activation[%d] * error[%d]\n", i, j-1, j);
            m_print(delta_W);
            #endif

            m_padd(tvals->layerouts[j].cum_d_W, delta_W); // accumulate d_W

            #ifdef DEBUG
            printf("%d:d_W #2 - cumulative d_W\n", i);
            m_print(tvals->layerouts[j].cum_d_W);
            #endif

            m_free(delta_W);
            m_free(tvals->layerouts[j-1].output);
            m_free(tvals->layerouts[j].error);

        }

        #ifdef DEBUG
        printf("Finished d_W accumulation #%d...\n", i);
        #endif

    }

    #ifdef DEBUG
    printf(">>> ------------------------------ <<<\n");
    printf(">>> FINISHED D_W ACCUM FOR BATCH!! <<<\n");
    printf(">>> ------------------------------ <<<\n");
    #endif

    //accumulated d_W in cum_d_W, time to
    //update the weights and thresholds by average
    for (int i = 1; i < net->num_layers; i++){

        //average cumulative deltas by # in epoch and mult by learning rate
        m_smulti(tvals->layerouts[i].cum_d_W, (learning_rate / input->num_rows));

        m_padd(tvals->layerouts[i].cum_d_W, tvals->layerouts[i].d_W); //add momentum * d_W_prev

        //store 'average' cum_d_W into d_W
        m_free(tvals->layerouts[i].d_W);
        tvals->layerouts[i].d_W = m_copy(tvals->layerouts[i].cum_d_W);

        m_smulti(tvals->layerouts[i].cum_d_W, 0.0); //reset cum_d_W to zero
        m_psub(net->layers[i].W, tvals->layerouts[i].d_W); //apply d_W to W

        m_smulti(tvals->layerouts[i].d_W, net->momentum); //apply momentum to d_W with p_multi

        #ifdef DEBUG
        printf("LAYER UPDATE--- layer %d\n Weights:\n", i);
        m_print(net->layers[i].W);
        printf("--              layer %d\n new (d_W * momentum) [added to next learningrate/minibatchsize * cum_d_W]:\n", i);
        m_print(tvals->layerouts[i].d_W);
        #endif

    }

    //cool our learning rate for next batch
    if (net->cooling_rate > 1){
        learning_rate /= net->cooling_rate;
    } else {
        fprintf(stderr, "nn_train: cooling rate must be >1 to effectively reduce learning_rate, not applied here\n");
    }

    printf("batch %d \tmax error at output:%f\t learning_rate:%f\n", batch++, batch_max_error, learning_rate);
    return batch_max_error;
}



void nn_train_txt(char *txtfile, net_t *net, int batch_size, int num_epochs, double max_error_tol){
//read a textfile for training our neural net and trains it
//times out after num_epochs is hit or max_error is less than max_err_tol

    matrix_t *input = m_init(batch_size, net->layers[0].size);
    matrix_t *goal = m_init(batch_size, net->layers[net->num_layers-1].size);

    FILE *ptr_file;
    size_t read;
    size_t len;
    char *line = malloc(20*sizeof(char));

    if (batch_size <= 0){
        fprintf(stderr, "nn_train_txt: Batch size must be >= 0\n");
        return;
    }

    int curr_row;

    double max_error = 999999999;
    int epoch = 0;

    while(max_error > max_error_tol && epoch < num_epochs){

        curr_row = 0;
        len = 0;

        ptr_file  = fopen(txtfile, "r");
        if (!ptr_file){
            fprintf(stderr, "nn_train_txt: Failed to open training text\n");
            return;
        }

        //iterate through file 1 full time for an epoch
        while ((read = getline(&line, &len, ptr_file)) != -1) {

                if(line[0] == line[1] && strcmp(&line[0],"/")) continue;
                if(read <= 1) continue;

                char *tofree = line;
                char *token, *val;
                double *cast_val = malloc(sizeof(double));

                int in_complete = 0;
                int ind = 0;
                while ((token = strsep(&line, ",")) != NULL) {
                    //splits on ','- seperate input/output vals
                    char *inner = token;
                    while ((val = strsep(&inner," \t")) != NULL) {
                        //splits on ' \t'- seperates into vals

                        if (sscanf(val, "%lf", cast_val) > 0){
                            //got a value we could parse

                            if (in_complete){
                                //write to goal matrix
                                if (ind >= goal->num_cols) {
                                    fprintf(stderr, "nn_train_txt: too many goal values for matrix\n");
                                    return;
                                }
                                goal->vals[curr_row][ind] = *cast_val;


                            } else{
                                //write to input matrix
                                if (ind >= input->num_cols) {
                                    fprintf(stderr, "nn_train_txt: too many input values for matrix\n");
                                    return;
                                }
                                input->vals[curr_row][ind] = *cast_val;
                            }

                            ind++;
                        }
                    }

                    //check we match the right length of neural net in/out
                    if (in_complete == 0 && ind != input->num_cols) {
                        fprintf(stderr, "nn_train_txt: Not enough values to fill input matrix\n");
                        return;
                    }
                    if (in_complete == 1 && ind != goal->num_cols) {
                        fprintf(stderr, "nn_train_txt: Not enough values to fill goal matrix\n");
                        return;
                    }

                    //switch from input to goal array
                    in_complete = 1;
                    ind = 0;
                }

                free(tofree);

                if (++curr_row == batch_size){
                    //call training for batch
                    max_error = nn_train(net, input, goal);
                    curr_row = 0;
                }


        }

        fclose(ptr_file);

        printf("--- EPOCH %d complete | error:%f ---\n\n", epoch, max_error);
        epoch++;
    }

    if (line) free(line);

}



matrix_t *nn_recall(net_t *net, matrix_t *input){
//use a neural network to predict an output given input

    //check input is right size
    if (input->num_cols != net->layers[0].size){
        fprintf(stderr, "nn_recall: num_cols of input don't match first layer of network.\n");
        return NULL;
    }
    if (input->num_rows != 1){
        fprintf(stderr, "nn_recall: num_rows of input should be one.\n");
        return NULL;
    }

    trainingvals_t *tvals = nn_init_trainingvals(net);
    tvals->layerouts[0].output = m_copy(input);

    //forward pass
    for (int i = 1; i < net->num_layers; i++) {

        tvals->layerouts[i].output = m_multi(tvals->layerouts[i-1].output, net->layers[i].W);
        m_padd(tvals->layerouts[i].output, net->layers[i].thresh);
        m_apply(tvals->layerouts[i].output, &sigmoid);
    }

    //free memory
    matrix_t *result = m_copy(tvals->layerouts[net->num_layers - 1].output);
    nn_trainingvals_free(tvals);

    return result;
}



int main(){

    const int num_layers = 4;
    int layer_sizes[num_layers] = {3,10,10,2};
    double momentum = 1;
    double learning_rate = 10;
    double cooling_rate = 1.01;

    //initialize the net
    net_t *n = nn_init(num_layers, layer_sizes, momentum, learning_rate, cooling_rate);

    //initialize random weights and biases for net
    nn_randomize_weights(n, 1, -1);

    nn_print(n);

    //then call training
    //nn_train_txt("./trainingsamples/training.txt", n, 2, 10, -1);
    nn_train_txt("./trainingsamples/training.txt", n, 2, 100, .000001);

    nn_print(n);

    //now test recall for samples in text
    matrix_t *input = m_init(1, 3);

    input->vals[0][0] = -0.4;
    input->vals[0][1] = -0.1;
    input->vals[0][2] = 0;
    printf("Results for [-.4, -.1, 0] (should be 1 0):\t");
    m_print(nn_recall(n, input));

    input->vals[0][0] = 0.5;
    input->vals[0][1] = 0;
    input->vals[0][2] = 0;
    printf("Results for [0.5, 0, 0] (should be 0 1):\t");
    m_print(nn_recall(n, input));

    input->vals[0][0] = 0.3;
    input->vals[0][1] = -0.1;
    input->vals[0][2] = -0.2;
    printf("Results for [0.3, -0.1, -0.2] (should be 0 1):\t");
    m_print(nn_recall(n, input));

    input->vals[0][0] = -1;
    input->vals[0][1] = -0.4;
    input->vals[0][2] = 0.5;
    printf("Results for [1, -0.4, 0.5] (should be 1 0):\t");
    m_print(nn_recall(n, input));
}


