	.arm
	.text	
	.global cost_cal
cost_cal:
	push	{r4-r6}
	ldr		r12,[sp,#12]	@hsumSub
	ldr 	r4,[sp,#16]		@C
	ldr		r5,[sp,#20]		@Count
	ldr		r6,[sp,#24]		@hsumAdd[x+d]
	lsr		r5,r5,#3		@Count/8
.loop:	
	vld1.16	{q0},[r0]!	@pixAdd[d]
	vld1.16	{q1},[r1]!	@pixSub[d]
	vld1.16	{q2},[r2]!	@hsumAdd[x-D+d]
	vqadd.s16 q3,q0,q2	
	vqsub.s16 q3,q3,q1	@hsumAdd[x+d]
	vst1.16	{q3},[r6]!	@hsumAdd[x+d]
	vld1.16	{q8},[r3]!	@Cprev[x+d]
	vld1.16	{q9},[r12]!	@hsumSub[x+d]
	vqadd.s16 q3,q3,q8
	vqsub.s16 q3,q3,q9
	vst1.16	{q3},[r4]!
	subs	r5,r5,#1
	bne		.loop
	pop           {r4-r6}
	bx            lr
	.end
	
