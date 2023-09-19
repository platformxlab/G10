Model       :   T_Identifier '(' LayerList ')'      
            ;

LayerList   :   LayerList Layer     
            |   Layer               
            ;

Layer       :   BlockLayer           
            |   SeqBlockLayer        
            ;

BlockLayer  :   '(' T_Identifier ')' ':' Block      
            ;

SeqBlockLayer:  '(' T_Identifier ')' ':' T_Sequential '(' SeqBlockList ')'  
            ;

SeqBlockList:   SeqBlockList SeqBlock  
            |   SeqBlock               
            ;

Block       :   UserBlock              
            |   OperatorBlock           
            ;

UserBlock   :   T_Identifier '(' LayerList ')'
            ;

OperatorBlock:  Operatorr        
            ;

SeqBlock    :   '(' T_IntConstant ')' ':' Block     
            |   '(' T_Identifier ')' ':' Block      
            ;

Operatorr   :   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_padding '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_bias '=' T_BoolConstant ')' 
            |   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_bias '=' T_BoolConstant ')'
            |   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_padding '=' '(' T_IntConstant ',' T_IntConstant ')' ')'
            |   T_ReLU '(' T_inplace '=' T_BoolConstant ')' 
            |   T_MaxPool2d '(' T_kernel_size '=' T_IntConstant ',' T_stride '=' T_IntConstant ',' T_padding '=' T_IntConstant ',' T_dilation '=' T_IntConstant ',' T_ceil_mode '=' T_BoolConstant ')'
            |   T_AdaptiveAvgPool2d '(' T_output_size '=' '(' T_IntConstant ',' T_IntConstant ')' ')'
            |   T_Linear '(' T_in_features '=' T_IntConstant ',' T_out_features '=' T_IntConstant ',' T_bias '=' T_BoolConstant ')'
            |   T_Dropout '(' T_p '=' T_DoubleConstant ',' T_inplace '=' T_BoolConstant ')'
            |   T_BatchNorm2d '(' T_IntConstant ',' T_eps '=' T_DoubleConstant ',' T_momentum '=' T_DoubleConstant ',' T_affine '=' T_BoolConstant ',' T_track_running_stats '=' T_BoolConstant ')'
            ;
