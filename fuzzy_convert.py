#!/usr/bin/python
#-*-encoding:utf-8-*-
class FuzzyConvert:
    def __init__(self,angle_axis_step = 10.0 , distance_axis_step = 0.5, cte_axis_step = 0.5):
        self.angle_axis_step=angle_axis_step
        self.distance_axis_step=distance_axis_step
        self.cte_axis_step=cte_axis_step
        self.angle_axis = []
        self.distance_axis = []
        self.cte_axis = []
        self.cet_fuzzy=['NB4','NB3','NB2','NB','NM','NS','ZO','PS','PM','PB','PB2','PB3','PB4']
        for i in range(-3,4):
            self.angle_axis.append( i * angle_axis_step)
            self.distance_axis.append( i * distance_axis_step)
        for i in range(-6,7):
            self.cte_axis.append( i * cte_axis_step)
        self.convert_mat = [
                                 ['NB4','NB3','NB2','NB','NM','NS','ZO'],
                                 ['NB3','NB2','NB','NM','NS','ZO','PS'],
                                 ['NB2','NB','NM','NS','ZO','PS','PM'],
                                 ['NB','NM','NS','ZO','PS','PM','PB'],
                                 ['NM','NS','ZO','PS','PM','PB','PB2'], 
                                 ['NS','ZO','PS','PM','PB','PB2','PB3'],
                                 ['ZO','PS','PM','PB','PB2','PB3','PB4']
                           ]
    def find_fuzzy_cte(self,val):
        for i in range(0,len(self.cet_fuzzy)):
            if val==self.cet_fuzzy[i]:
                return i
        return -1

    def cal_cte_level(self,angle_value,distance_value):
        angle_score=[]
        distance_score=[]
        cte_score=[]

        if angle_value<=self.angle_axis[0]:
            angle_score.append([0,1.0])
        elif  angle_value>=self.angle_axis[len(self.angle_axis)-1]:
            angle_score.append([len(self.angle_axis)-1,1.0])
        else:
            for i in range(1,len(self.angle_axis)):
                if (angle_value<=self.angle_axis[i]):

                    left_bound=(self.angle_axis[i]-angle_value)/self.angle_axis_step
                    right_bound=1.0-left_bound
                    angle_score.append([i-1,left_bound]) 
                    angle_score.append([i,right_bound]) 
                    break
        if distance_value<=self.distance_axis[0]:
            distance_score.append([0,1.0])
        elif  distance_value>=self.distance_axis[len(self.distance_axis)-1]:
            distance_score.append([len(self.distance_axis)-1,1.0])
        else:
            for i in range(1,len(self.distance_axis)):
                if (distance_value<=self.distance_axis[i]):

                    left_bound=(self.distance_axis[i]-distance_value)/self.distance_axis_step
                    right_bound=1.0-left_bound
                    distance_score.append([i-1,left_bound]) 
                    distance_score.append([i,right_bound]) 
                    break
        for i in range(0,len(angle_score)):
            for j in range(0,len(distance_score)):
                convert_result = self.convert_mat[angle_score[i][0]][distance_score[j][0]]
                index = self.find_fuzzy_cte(convert_result)
                assert index>-1
                
                score = min(angle_score[i][1] , distance_score[j][1])
                cte_score.append([index,score])
        result=0.0        
        weight_sum=0.0
        for i in range(0,len(cte_score)):
            temp_cte=self.cte_axis[cte_score[i][0]]
            temp_weight=cte_score[i][1]
            result=result+temp_cte*temp_weight
            weight_sum=weight_sum+temp_weight
        result=result/weight_sum
        return result
