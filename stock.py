# 장애물 회피 게임 즉, 자율주행차:-D 게임을 구현합니다.
import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, minmax_scale

class Stock:
    def __init__(self, Data_name, day_length, show_stock=True):
        with open(Data_name, 'r') as f:
            ftmp = f.readlines()
            self.data = []
        for tmp in ftmp:
            self.data.append(tmp.replace(',','').split())
        self.data = np.array(self.data, np.float32)
            
        self.day_length = day_length
        self.max_day = len(self.data)
        self.mm_tracking = np.zeros([self.max_day,1])
        self.num_data = len(self.data[0]) +1
        #시가, 고가, 저가, 종가, 거래량, 개인순매수량- 6개
        self.start_price = self.data[day_length-1][0]
        self.high_price = self.data[day_length-1][1]
        self.low_price = self.data[day_length-1][2]
        self.end_price = self.data[day_length-1][3]
        self.volume = self.data[day_length-1][4]
        self.Indi_mount = self.data[day_length-1][5]

        self.buy_check = False
        self.current_reward = 0
        self.total_reward = 0
        self.time_step = 0
        self.current_point= self.day_length
        self.buy_point = []
        self.cell_point = []
        self.buy_price = 0
        self.cell_price = 0
        self.show_stock = show_stock
        self.total_test = 0
        self.trade_count = 0
        
        if self.show_stock:
            self.fig, self.ax = plt.subplots()
            # 화면을 닫으면 프로그램을 종료합니다.
            self.fig.canvas.mpl_connect('close_event', exit)
            #self.fig, self.axis = self._prepare_display(self.day_length)
            
    def _prepare_display(self, day_length):
        #주가 그래프와 매수 매도 포인트.
        x_length = day_length + 20
        x_max = np.max([self.current_point, x_length])
        x_min = np.max([0, self.current_point - x_length])
        x_state = self.current_point-1
        p_data = np.array(self.data[x_min:x_max], np.float32)
        p_data = p_data[:,3]
        y_tmp = np.array(self.data[x_min:self.current_point], np.float32)
        y_max = 1.1*max(y_tmp[:,1])
        y_min = 0.9*min(y_tmp[:,2])
        
        x_axis = [ax for ax in range(x_min, x_max)]
        self.ax.axis((x_min, x_max, y_min, y_max))
        self.ax.plot(x_axis,p_data)
        self.ax.annotate(r'$current=%f$' % self.end_price, xy=(x_state, self.end_price), xycoords='data',
                     xytext=(+25, +20), textcoords='offset points', fontsize=12,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        self.ax.plot([x_state,x_state],[y_min,self.end_price], color ='black', linewidth=1.5, linestyle="--")
        self.ax.plot([x_min,x_max],[self.end_price,self.end_price], color ='black', linewidth=1.0, linestyle="--")
        
        if(self.buy_check):
            self.ax.plot([x_min,x_max],[self.buy_price,self.buy_price], color ='red', linewidth=1.0, linestyle="--")
            buy_state = np.array(self.buy_point[-1], np.float32)
            if(buy_state > x_max-x_length-1):
                self.ax.annotate(r'$buy=%f$' % self.buy_price, xy=(x_state, self.buy_price), xycoords='data',
                    xytext=(+25, +20), textcoords='offset points', fontsize=12,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                self.ax.plot([buy_state,buy_state],[y_min,self.buy_price], color ='red', linewidth=1.5, linestyle="--")
        
#        plt.draw()
        # 게임을 진행하며 화면을 업데이트 할 수 있도록 interactive 모드로 설정합니다.
#        plt.ion()
#        plt.show()
        
    def reset(self):
        """주가 data reset"""
            #시가, 고가, 저가, 종가, 거래량, 개인순매수량- 6개
        self.current_point = self.day_length
        self.mm_tracking = np.zeros(self.max_day)
        self.start_price = self.data[self.current_point-1][0]
        self.high_price = self.data[self.current_point-1][1]
        self.low_price = self.data[self.current_point-1][2]
        self.end_price = self.data[self.current_point-1][3]
        self.volume = self.data[self.current_point-1][4]
        self.Indi_mount = self.data[self.current_point-1][5]
        self.buy_check = False
        self.current_reward = 0
        self.total_reward = 1
        self.time_step = 0
        self.buy_point = []
        self.cell_point = []
        self.buy_price = 0
        self.cell_price = 0
        self.total_test += 1

        if self.show_stock:
            self.draw_screen()

        return self._get_state()    
        
    def _get_state(self):
        """현재 주가 상태를 가져옴.
        """
        state = self.data[self.current_point-self.day_length:self.current_point].copy()
        mm_line = self.mm_tracking[self.current_point-self.day_length:self.current_point].copy()
        tmp_price = state[:, 0:4]
        flat_price = np.reshape(tmp_price, [1,-1])
        norm_val = normalize(flat_price, norm = 'max')
        norm_price = np.reshape(norm_val, [self.day_length, -1])
        
        volumes = state[:, 4]
        vol = np.reshape(volumes, [1,-1])
        Tvol = sum(vol.T)
        norm_volume = vol/Tvol
        
        Indi = state[:, 5]
        norm_indi = 2*minmax_scale(Indi)-1  

        state[:,0:4] = norm_price
        state[:,4] = norm_volume
        state[:,5] = norm_indi
        mm_line = np.reshape(mm_line, [self.day_length,-1])
        state = np.append(state, mm_line, axis=1)

        return state    
    
    def draw_screen(self):
        self.ax.cla()
        title = " Avg. Reward: %d Reward: %d Total Test: %d" % (
                        self.total_reward / self.total_test,
                        self.current_reward,
                        self.total_test)

        # self.axis.clear()
        self.ax.set_title(title, fontsize=12)
        #self.fig.canvas.draw()
        self. _prepare_display(self.day_length)
        plt.pause(0.0001)
       # self.fig, self.axis = self._prepare_display(self.day_length)

    def _update_stock(self, move):
        """액션에 매매를 시작합니다.
        """
        #print([self.current_point, self.max_day])
        self.start_price = self.data[self.current_point][0]
        self.high_price = self.data[self.current_point][1]
        self.low_price = self.data[self.current_point][2]
        self.end_price = self.data[self.current_point][3]
        self.volume = self.data[self.current_point][4]
        self.Indi_mount = self.data[self.current_point][5]

        if(move == 1):
            if(self.buy_check==False):
                self.buy_check = True
                self.buy_point.append(self.current_point)
                self.mm_tracking[self.current_point] = 1
                self.buy_price = self.start_price
                self.current_reward = self._buy_reward()
            else:
                #self.current_reward -=1  
                self.mm_tracking[self.current_point] = self.end_price/self.buy_price
            
        elif (move == 0):
            if(self.buy_check):
                self.mm_tracking[self.current_point] = self.end_price/self.buy_price
                
                if(self.data[self.current_point][3] > self.data[self.current_point+1][3]):
                    self.time_step = self.time_step + 1
                    #self.current_reward -= self._time_reward(self.time_step) + 0.1*(self.end_price / self.buy_price - 1)
                    self.current_reward -= self._time_reward(self.time_step)
                else:
                    self.current_reward = 0
            else:
                if(self.data[self.current_point][3] < self.data[self.current_point+1][3]):
                    self.time_step = self.time_step + 1
                    #self.current_reward -= self._time_reward(self.time_step) + 0.1*(self.end_price / self.buy_price - 1)
                    self.current_reward -= self._time_reward(self.time_step)
                else:
                    self.current_reward = 0
                
                #self.current_reward = 0
                self.mm_tracking[self.current_point] = 0
                self.time_step = 0  
                #print([self.start_price ,self.data[self.current_point][0]])
                #self.current_reward -= 0.05*(self.start_price / self.data[self.current_point-1][0] - 1)
                
        elif(move == -1):
            if(self.buy_check):
                self.buy_check = False
                self.mm_tracking[self.current_point] = self.start_price/self.buy_price
                self.current_reward = self._cell_reward()+ self.start_price/self.buy_price -1
                self.total_reward *= self.start_price/self.buy_price
                self.time_step = 0  
                self.cell_point.append(self.current_point)
                self.cell_price = self.start_price
            else:
                #self.current_reward -=1
                self.mm_tracking[self.current_point] = 0				
        
        self.current_point = self.current_point + 1

    def _time_reward(self, time_step):
        return 0.2*time_step**2/200
    
    def _buy_reward(self):
        min_p = np.argmin(self.data[self.current_point+1:self.current_point+5][3])
        max_p = np.argmax(self.data[self.current_point+1:self.current_point+5][3])
        if(max_p < min_p):
            return 10*(self.data[self.current_point+max_p+1][0] / self.buy_price - 1)
        else:
            return 10*(self.data[self.current_point+min_p+1][0] / self.buy_price - 1)

    def _cell_reward(self):
        min_p = np.argmin(self.data[self.current_point+1:self.current_point+5][3])
        max_p = np.argmax(self.data[self.current_point+1:self.current_point+5][3])
        if(max_p > min_p):
            return 10*(1-self.data[self.current_point+min_p][3] / self.data[self.current_point][3])
        else:
            return 10*(1-self.data[self.current_point+max_p][3] / self.data[self.current_point][3])

    def _is_gameover(self):
        # 매매 종료
        if (self.current_point >= self.max_day-5):
            if(len(self.cell_point) >0):
                self.total_reward *= self.end_price/self.buy_price
            else:
                self.current_reward = -2
            return True
        #elif(self.current_reward<-0.1):
        #    self.current_reward = -1
        #    return True
        else:
            return False
        
    def step(self, action):
        # action: 0: 매도, 1: 유지, 2: 매수
        # action - 1 을 하여, 좌표를 액션이 0 일 경우 -1 만큼, 2 일 경우 1 만큼 옮깁니다.
        self._update_stock(action - 1)
        self.trade_count +=1
        gameover = self._is_gameover()

       # if gameover:
        #    reward = -2
        #else:
        #if gameover:
        #    reward = self.current_reward
        #else:
        #    reward = 0
        reward = self.current_reward
        total_reward = self.total_reward
		
        if self.show_stock:
            self.draw_screen()

        return self._get_state(), reward, total_reward, gameover  
        