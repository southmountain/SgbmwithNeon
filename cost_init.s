	.arm
	.text	
	.global cost_init
cost_init:
	push	{r4-r7}
	ldr		r12,[sp,#16]	@v
	ldr		r4,[sp,#20]		@v0
	ldr		r5,[sp,#24]		@v1
	ldr		r6,[sp,#28]		@cost
	ldr		r7,[sp,#32]		@D
	lsr		r7,r7,#4 
	vdup.8	q0,r0		@u
	vdup.8	q1,r1		@u0
	vdup.8	q2,r2		@u1
	vdup.16	 q12,r3		@(-1)*diff_scale

.loop:	
	vld1.8	{q3},[r12]!	@v
	vld1.8	{q8},[r4]!	@v0
	vld1.8	{q9},[r5]!	@v1
	vqsub.s8	q9,q0,q9	@u-v1
	vqsub.s8	q8,q8,q0	@v0-u
	vmax.s8		q8,q8,q9	@c0=max(u-v1,v0-u)
	vqsub.s8	q10,q3,q2	@v-u1
	vqsub.s8	q11,q1,q3	@u0-v
	vmax.s8		q10,q10,q11	@c1=max(v-u1,u0-v)
	vmin.s8		q3,q8,q10	@min(c0,c1)
	vmovl.s8	q10,d6
	vmovl.s8	q11,d7
	vshl.s16	q10,q10,q12
	vshl.s16	q11,q11,q12
	vld1.16		{q3},[r6]
	vqadd.s16	q10,q10,q3
	vst1.16		{q10},[r6]!
	vld1.16		{q3},[r6]
	vqadd.s16	q11,q11,q3
	vst1.16		{q11},[r6]!
	subs	r7,r7,#1
	bne			.loop
	
	pop           {r4-r7}
	bx            lr
	.end


