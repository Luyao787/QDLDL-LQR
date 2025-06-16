#pragma once

#include "data_types.h"

void qdldl_cleanup(QDLDLData* data)
{
    if (data) {
        if (data->Lp) free(data->Lp);
        if (data->Li) free(data->Li);
        if (data->Lx) free(data->Lx);
        if (data->D) free(data->D);
        if (data->Dinv) free(data->Dinv);

        if (data->etree) free(data->etree);
        if (data->Lnz) free(data->Lnz);
       
        if (data->iwork) free(data->iwork);
        if (data->bwork) free(data->bwork);
        if (data->fwork) free(data->fwork);

        if (data->x) free(data->x);
        
        free(data);
    }
}

QDLDLData* qdldl_setup(CscMatrix* K)
{
    QDLDLData* data = (QDLDLData*) calloc(1, sizeof(QDLDLData));
    if (!data) {
        return 0;
    }

    QDLDL_int n = K->n; // Number of columns

    data->Ln = n;

    data->etree = (QDLDL_int*) malloc(n * sizeof(QDLDL_int));
    data->Lnz   = (QDLDL_int*) malloc(n * sizeof(QDLDL_int));

    data->Lp    = (QDLDL_int*) malloc((n + 1) * sizeof(QDLDL_int));
    data->D     = (QDLDL_float*) malloc(n * sizeof(QDLDL_float));
    data->Dinv  = (QDLDL_float*) malloc(n * sizeof(QDLDL_float));

    data->iwork = (QDLDL_int*) malloc(3 * n * sizeof(QDLDL_int));
    data->bwork = (QDLDL_bool*) malloc(n * sizeof(QDLDL_bool));
    data->fwork = (QDLDL_float*) malloc(n * sizeof(QDLDL_float));

    data->sumLnz = QDLDL_etree(n, K->p, K->i, data->iwork, data->Lnz, data->etree);

    if (data->sumLnz < 0) {
        std::cerr << "Error in QDLDL_etree: " << std::endl;
        qdldl_cleanup(data);
        return 0;
    }
    
    data->Li = (QDLDL_int*) malloc(data->sumLnz * sizeof(QDLDL_int));
    data->Lx = (QDLDL_float*) malloc(data->sumLnz * sizeof(QDLDL_float));

    data->x  = (QDLDL_float*) malloc(n * sizeof(QDLDL_float));

    return data;
}

