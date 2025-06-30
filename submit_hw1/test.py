'''
(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.10704800300000006
canny_50_100_elapsed_time : 0.0021240939999997988
canny_0_0_elapsed_time : 0.007362290000000105
canny_400_400_elapsed_time : 0.0005147469999999821
bilater_default_param_elapsed_time : 0.0027159100000000436
bilater_elapsed_time : 0.004172491000000056

(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.12970442000000004
canny_50_100_elapsed_time : 0.258593257
canny_0_0_elapsed_time : 0.2279441550000001
canny_400_400_elapsed_time : 0.0029527600000001097
bilater_default_param_elapsed_time : 0.0018168579999999768
bilater_elapsed_time : 0.004173997000000096

(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.004111230999999993
canny_50_100_elapsed_time : 0.003699625000000095
canny_0_0_elapsed_time : 0.007407985000000172
canny_400_400_elapsed_time : 0.0002669870000000518
bilater_default_param_elapsed_time : 0.07272809000000002
bilater_elapsed_time : 0.04214134099999978

(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.0049828450000000135
canny_50_100_elapsed_time : 0.008601447000000206
canny_0_0_elapsed_time : 0.016559173000000094
canny_400_400_elapsed_time : 0.0002325480000000546
bilater_default_param_elapsed_time : 0.05717069100000005
bilater_elapsed_time : 0.11949308800000047

(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.1668602669999999
canny_50_100_elapsed_time : 0.003071684000000019
canny_0_0_elapsed_time : 0.017527756999999866
canny_400_400_elapsed_time : 0.0004897569999999796
bilater_default_param_elapsed_time : 0.08841403999999997
bilater_elapsed_time : 0.05442019100000017

(cv_hw) sivvon@sivvon-B460MDS3HV2:~/Desktop/CV_Class_HW$ /home/sivvon/anaconda3/envs/cv_hw/bin/python /home/sivvon/Desktop/CV_Class_HW/cv_hw_01_problem_01_sivvon_yoon.py
2d_gaussian_elapsed_time : 0.0035501260000001533
canny_50_100_elapsed_time : 0.4595137180000002
canny_0_0_elapsed_time : 0.14201058999999994
canny_400_400_elapsed_time : 0.0002976070000002551
bilater_default_param_elapsed_time : 0.4125922270000002
bilater_elapsed_time : 0.0028328840000000355
'''
# g = 0.10704800300000006 #1
# g += 0.12970442000000004 #2
# g += 0.004111230999999993 #3
# g += 0.0049828450000000135 #4
# g += 0.1668602669999999 #5
# g += 0.0035501260000001533 # 6
# g = g/6
# print(g)

harris_1 = 0.2335187970000001 / 224163
harris_2 = 0.022731025999999988 / 158412

fast_1 = 3.1689999999695573e-06 / 2829
fast_2 = 3.277000000023733e-06 / 502

print(f'harris_1 is {harris_1} and fast_1 is {fast_1} than harris is slower than  FAST --> ',harris_1 > fast_1)
print(f'harris_2 is {harris_2} and fast_2 is {fast_2} than harris is slower than  FAST --> ',harris_2 > fast_2)