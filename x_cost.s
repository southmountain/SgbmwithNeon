	.arm
	.text	
	.global x_cost
x_cost:
		push		{r4-r6}
		ldr			r12,[sp,#12]	@Sp
		ldr			r4,[sp,#16]	@d8
		ldr			r5,[sp,#20] @D
		ldr			r6,[sp,#24] @minvalue
		lsr			r5,r5,#3
		vdup.16       q0,r0       @delta0
		 mov           r0,#200     
		vdup.16       q1,r0       @p1
		mov         r0,#255
		lsl 	    r0,r0,#7
		add         r0,r0,#127	
		vdup.16			q9,r0    @minL0
		vdup.16			q10,r0	  @mins
		mov			r0,#-1
		vdup.16			q11,r0	  @bestdisp
		vld1.16			{q12},[r4]	@d8
		mov			r0,#8
		vdup.16			q13,r0		@_8
		
.loop:		
		vld1.16       {q2},[r1]!    @Cp[d]
		vld1.16       {q3},[r2]    @Lr_p0[d]
		vmin.s16	   q8,q3,q0		@min(delta0,Lr_p0[d])
		sub 			r2,r2,#2
		vld1.16			{q3},[r2]	@Lr_p0[d-1]
		vqadd.s16		q3,q3,q1	@Lr_p0[d-1]+P1
		vmin.s16		q8,q8,q3	@min(delta0,Lr_p0[d],Lr_p0[d-1]+P1)
		add				r2,r2,#4
		vld1.16			{q3},[r2]	@Lr_p0[d+1]
		vqadd.s16		q3,q3,q1	@Lr_p0[d+1]+P1
		vmin.s16		q8,q8,q3	@min(delta0,Lr_p0[d],Lr_p0[d-1]+P1,Lr_p0[d+1]+P1)
		vqsub.s16		q8,q8,q0	@minus delta0
		vqadd.s16		q8,q8,q2	@L0
		vst1.16			{q8},[r3]!	@Lr_p[d]
		vmin.s16		q9,q9,q8	@min(minL0,L0)
		vld1.16			{q2},[r12]	@Sp[d]
		vqadd.s16		q2,q2,q8	@Sp[d]=Sp[d]+L0
		vst1.16			{q2},[r12]! 
		vcgt.s16		q3,q10,q2	@mask
		vmin.s16		q10,q10,q2	@min(mins,sp[d])
		veor			q8,q11,q12	@xor(bestdisp,d8)
		vand			q8,q8,q3
		veor			q11,q11,q8	@bestdisp
		vqadd.s16		q12,q12,q13	@d8+8
		add				r2,r2, #14
		subs				r5,r5,#1
		bne         	.loop
		
		vmin.s16		d0,d18,d19
		vmin.s16		d1,d20,d21
		vtrn.16			d0,d1
		vmin.s16		d0,d0,d1
		vshr.s64		d2,d0,#32
		vmin.s16		d0,d2,d0
		vst1.16			{d0},[r6]!
		
		vdup.16			q3,d0[1]
		vceq.s16		q3,q10,q3	@qs
		vand			q2,q11,q3
		vst1.16			{q2},[r6]!
		pop           {r4-r6}
		bx            lr
		.end
