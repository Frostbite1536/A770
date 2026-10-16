-- Hand-written synthetic timings, never A770 benchmark results.
CREATE TABLE test_backend_ops (op_name TEXT, op_params TEXT, test_mode TEXT, supported INTEGER, time_us REAL);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=1,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 30);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=16,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 60);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q8_0,type_b=f32,m=1024,n=1,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 40);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q6_K,type_b=f32,m=14336,n=16,k=4096,bs=[1,1],nr=[1,1]', 'perf', 0, NULL);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=unknown,type_b=f32,m=4096,n=1,k=4096,bs=[1,1],nr=[1,1]', 'perf', 1, 15);
INSERT INTO test_backend_ops VALUES ('MUL_MAT', 'type_a=q4_K,type_b=f32,m=4096,n=1,k=4096,bs=[1,1],nr=[1,1]', 'test', 1, 999);
INSERT INTO test_backend_ops VALUES ('ADD', 'type=f32', 'perf', 1, 1);
