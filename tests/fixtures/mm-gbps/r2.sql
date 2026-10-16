-- Hand-written second round: medians with r1 are 20 and 50 us.
CREATE TABLE test_backend_ops (op_name TEXT, op_params TEXT, test_mode TEXT, supported INTEGER, time_us REAL);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=1,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 10);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=16,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 40);
