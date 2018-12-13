#!/usr/bin/python
#-*-encoding:utf-8-*-
class PIDControl:
    def __init__(self, kp, ki, kd):
        self.Kp=kp
        self.Ki=ki
        self.Kd=kd
        self.i_error = 0.0;
        self.d_error = 0.0;
        self.p_error = 0.0;
        self.last1=0.0;
        self.last2=0.0;
    def UpdateError(self, cte):
        self.p_error = cte - self.last1;
        self.i_error = cte;
        self.d_error = cte - 2 * self.last1 + self.last2;
        self.last2 = self.last1;
        self.last1 = cte;

    def TotalError(self):
        return -( self.Kp *  self.p_error +  self.Ki *  self.i_error +  self.Kd *  self.d_error);

    def reflash(self):
        self.i_error = 0.0;
        self.d_error = 0.0;
        self.p_error = 0.0;
        self.last1 = 0.0;
        self.last2 = 0.0;