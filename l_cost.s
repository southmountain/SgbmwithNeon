    .arm
	.text	
	.global l_cost
l_cost:
        push     {r4-r9}
		ldr		r12,[sp,#24]  @lr_p0
		ldr     r4,[sp,#28]   @lr_p1
		ldr     r5,[sp,#32]   @Cp
		ldr     r6,[sp,#36]   @Sp
		ldr     r7,[sp,#40]   @minLr[0][xm]
		ldr		r8,[sp,#44]	  @D
		
		vpush           {q4}

		
		vdup.16       q0,r0       @delta0
		vdup.16       q1,r1      @delta1
		vdup.16       q2,r2      @delta2
		vdup.16       q3,r3      @delta3
		mov           r0,#200     
		vdup.16       q4,r0       @p1
		mov         r0,#255
		lsl 	    r0,r0,#7
		add         r0,r0,#127	
		vdup.16			q8,r0	@minL0
		vdup.16			q9,r0	@minL1
		vdup.16			q10,r0	@minL2
		vdup.16			q11,r0	@minL3
		add			r1,r8,#16	@D2
		mov			r9,#22
		mul			r9,r9,r1
		sub			r9,r9,#16	@22*D2-16
		lsl			r2,r1,#3	@NRD2
		sub			r3,r2,#1
		lsl			r3,r3,#1	@2*(NRD2-1)
		lsl			r1,r1,#1	@2*D2
		add			r0,r1,r3	@2*(NRD2-1+D2)
		lsl			r2,r0,#1	@4*(NRD2-1+D2)
		sub			r2,r2,#10	@4*NRD2+4*D2-14
		lsr			r8,r8,#3	@D/8
		
.loop:		
		vld1.16       {q12},[r5]!    @Cp[d]
		vld1.16			{q13},[r6]	@Sp[d] 
		
		vld1.16   	 {q14},[r12]   @lr_p0[d]
		sub          r12,r12,#2     
		vld1.16      {q15},[r12]   @lr_p0[d-1]
		vqadd.s16     q15,q15,q4   @lr_p0[d-1]+p1
		vmin.s16	q14,q14,q15	@min(lr_p0[d],lr_p0[d-1]+p1)
		add          r12,r12,#4    
		vld1.16      {q15},[r12]  @lr_p0[d+1]
		vqadd.s16	q15,q15,q4	@lr_p0[d+1]+p1
		vmin.s16	q14,q14,q15	@min(lr_p0[d],lr_p0[d-1]+p1,lr_p0[d+1],p1)
		vmin.s16      q14,q14,q0
		vqsub.s16     q14,q14,q0
		vqadd.s16     q14,q14,q12  @L0
		add			r12,r12,r3		@Lr_p[d]
		vst1.16			{q14},[r12]	@Lr_p[d]
		vmin.s16	q8,q8,q14	@min(minL0,L0)
		vqadd.s16	q13,q13,q14	@Sp[d]+L0

		vld1.16       {q14},[r4] @lr_p1[d]
		sub           r4,r4,#2  
		vld1.16       {q15},[r4]  @lr_p1[d-1]
		vqadd.s16	  q15,q15,q4	@lr_p1[d-1]+p1
		vmin.s16		q14,q14,q15	@min(lr_p1[d],lr_p1[d-1]+p1)
		add           r4,r4,#4
		vld1.16       {q15},[r4]  @lr_p1[d+1]
		vqadd.s16     q15,q15,q4   @lr_p1[d+1]+p1
		vmin.s16      q14,q14,q15 @min(lr_p1[d],lr_p1[d-1]+p1,lr_p1[d+1]+p1)
		vmin.s16      q14,q14,q1 
		vqsub.s16     q14,q14,q1
		vqadd.s16     q14,q14,q12 @L1
		add			r12,r12,r1
		vst1.16		{q14},[r12]	@Lr_p[d+D2]
		vmin.s16	q9,q9,q14	@min(minL1,L1)
		vqadd.s16	q13,q13,q14	@Sp[d]+L0+L1
		
		add			r4,r4,r0  
		vld1.16       {q14},[r4] @lr_p2[d]
		sub           r4,r4,#2
		vld1.16       {q15},[r4]  @lr_p2[d-1]
		vqadd.s16	q15,q15,q4	@lr_p2[d-1]+p1
		vmin.s16	q14,q14,q15	@min(lr_p2[d],lr_p2[d-1]+p1)
		add           r4,r4,#4
		vld1.16       {q15},[r4]  @lr_p2[d+1]
		vqadd.s16     q15,q15,q4   @lr_p2[d+1]+p1
		vmin.s16      q14,q14,q15  @min(lr_p2[d],lr_p2[d-1]+p1,lr_p2[d+1]+p1)
		vmin.s16      q14,q14,q2
		vqsub.s16     q14,q14,q2
		vqadd.s16     q14,q14,q12 @L2
		add			r12,r12,r1
		vst1.s16	{q14},[r12]		@Lr_p[d+2*D2]
		vmin.s16	q10,q10,q14		@min(minL2,L2)
		vqadd.s16	q13,q13,q14		@Sp[d]+L0+L1+L2
	
	
		add			r4,r4,r0 
		vld1.16       {q14},[r4] @lr_p3[d]
		sub           r4,r4,#2
		vld1.16       {q15},[r4]  @lr_p3[d-1]
		vqadd.s16	q15,q15,q4		@lr_p3[d-1]+p1
		vmin.s16	q14,q14,q15		@min(lr_p3[d],lr_p3[d-1]+p1)
		add           r4,r4,#4
		vld1.16       {q15},[r4]  @lr_p3[d+1]
		vqadd.s16     q15,q15,q4   @lr_p3[d+1]+p1
		vmin.s16      q14,q14,q15  @min(lr_p3[d],lr_p3[d-1]+p1,lr_p3[d+1]+p1)
		vmin.s16      q14,q14,q3
		vqsub.s16     q14,q14,q3
		vqadd.s16     q14,q14,q12 @L3
		add			r12,r12,r1
		vst1.16		{q14},[r12]	@Lr_p[d+3*D2]
		vmin.s16	q11,q11,q14	@min(minL3,L3)
		vqadd.s16	q13,q13,q14	@Sp[d]+L0+L1+L2+L3
		vst1.16		{q13},[r6]!	@Sp[d]
		sub			r12,r12,r9	@Lr_p[d+3*D2]->Lr_p0[d+8]
		sub			r4,r4,r2	@Lr_p3[d+1]->Lr_p1[d+8]
		subs		r8,r8,#1
		bne			.loop
		
		
		
		vtrn.16       q8,q9        @L0,L1
		vtrn.16       q10,q11        @L2,L3
		vmin.s16      q8,q8,q9     
		vmin.s16      q10,q10,q11
		vmin.s16      d0,d16,d17
		vmin.s16      d1,d20,d21
		vshr.s64      q1,q0,#32
		vmin.s16		q0,q0,q1
		vtrn.32		d0,d1		@minL0,minL1,minL2,minL3
		vst1.16		{q0},[r7]
		
		
		vpop           {q4}
		pop           {r4-r9}
		bx            lr
		.end
