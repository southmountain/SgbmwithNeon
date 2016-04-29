	.arm
	.text	
	.global cost_final
cost_final:

		vdup.8	q0,r0	@u
		lsr		r3,r3,#4	@D/16
.loop:
		vld1.8	{q1},[r1]!	@v
		vabd.s8	q2,q0,q1	@abs(u-v)
		vmovl.s8	q3,d4
		vmovl.s8	q8,d5
		vld1.16	{q1},[r2]
		vqadd.s16	q1,q1,q3
		vst1.16	{q1},[r2]!
		vld1.16	{q2},[r2]
		vqadd.s16	q2,q2,q8
		vst1.16	{q2},[r2]!
		subs	r3,r3,#1
		bne		.loop

		bx            lr
		.end
		
		
