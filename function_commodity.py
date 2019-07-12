# important!!!! estimate for one slideing window parameters
def estimate_for_parameters(Y_100,theta_old,sigma_old):#,theta_init):

    def estimate_for_Y( L_tilta, Y_window, L_used_for_Y,delta1):

        def get_A(L_tilta, Y_window):
            result = 0
            for i in range(30):
                temp = np.sum(np.square(Y_window[79*i:79*(i+1)-1])) + 78*(L_tilta[i]**2)- 2*L_tilta[i]*(np.sum(Y_window[79*i:79*(i+1)-1]))
                result += temp
            return result

        def get_B(L_tilta, Y_window):
            result = 0
            for i in range(30):
                temp1 = (-156)*(L_tilta[i]**2) + 2*L_tilta[i]*np.sum(Y_window[79*i+1:79*(i+1)])+2*L_tilta[i]*np.sum(Y_window[79*i:79*(i+1)-1])
                temp2 = 2*np.sum([Y_window[79*i+j]*Y_window[79*i+j+1] for j in range(78)])
                temp = temp1-temp2
                result += temp
            return result

        def get_C(L_tilta,Y_window):
            result = 0
            for i in range(30):
                temp = np.sum(np.square(Y_window[79*i+1:79*(i+1)])) + 78*(L_tilta[i]**2)- 2*L_tilta[i]*(np.sum(Y_window[79*i+1:79*(i+1)]))
                result += temp
            return result

        def get_D(L_used_for_Y):
            result = 0
            for i in range(1,len(L_used_for_Y),2):
                result += ((L_used_for_Y[i]-L_used_for_Y[i-1])/2)**2
            return result

        def get_F(L_used_for_Y):
            result = 0
            for i in range(1,len(L_used_for_Y),2):
                temp = L_used_for_Y[i+1]**2 + L_used_for_Y[i]**2/4 + L_used_for_Y[i-1]**2/4 - L_used_for_Y[i]*L_used_for_Y[i+1] \
                - L_used_for_Y[i+1]*L_used_for_Y[i-1] + L_used_for_Y[i]*L_used_for_Y[i-1]/2
                result += temp
            return result

        def get_E(L_used_for_Y):
            result = 0
            for i in range(1,len(L_used_for_Y),2):
                result += L_used_for_Y[i]**2/2 - L_used_for_Y[i-1]**2/2 - L_used_for_Y[i+1]*L_used_for_Y[i] + L_used_for_Y[i+1]*L_used_for_Y[i-1]
            return result

        def MLE_for_Y(x): # 传入要优化的参数 theta
            #delta = delta1/78 # delta1 comes from the estimate for L
            a = np.exp(-x*delta1/78)
            sigma_theta = np.sqrt(2/77/30*(x/(1-a**2)*(A*a**2+B*a+C)-x/(1-a**156)*(D*a**156+E*a**78+F)))
            #print(sigma_theta)
            f = -77*30*np.log(sigma_theta) - 77*30/2 * np.log((1-a**2)/x) - 77*30/2 * np.log(np.pi)\
            + 15*np.log((1-a**156)/(1-a**2)) - 77*15
            return -f


        A = get_A(L_tilta, Y_window)
        B = get_B(L_tilta, Y_window)
        C = get_C(L_tilta, Y_window)
        D = get_D(L_used_for_Y)
        E = get_E(L_used_for_Y)
        F = get_F(L_used_for_Y)



        #print(A,B,C,D,E,F)
        opt_min = 5
        for i in np.linspace(1,500,100):
            opt = scipy.optimize.minimize(MLE_for_Y, i , method='L-BFGS-B',args=(), jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
            #print(opt)
            theta_temp_Y = opt.x[0]
            opt_temp = opt.fun[0]
            if opt_temp < opt_min:
                opt_min = opt_temp
                theta_Y = theta_temp_Y

        a = np.exp(-theta_Y*delta1/78)
        sigma_Y = np.sqrt(2/77/30*(theta_Y/(1-a**2)*(A*a**2+B*a+C)-theta_Y/(1-a**156)*(D*a**156+E*a**78+F)))


        #print(theta_Y,sigma_Y)


        return theta_Y,sigma_Y

    def estimate_for_L(L_window,theta_old = 9, sigma_old = 0.2):

        def MLE(x):
        # x - list of (theta_L, sigma_L)
            N = 100
            sigma1_h = x[1]*np.sqrt((1-(np.e)**(-2*x[0]*delta1))/(2*x[0]))
            sigma2_h = x[1]*np.sqrt((1-(np.e)**(-2*x[0]*delta2))/(2*x[0]))
            f = -N/2*np.log(2*np.pi) - N*np.log(sigma1_h) - 1/(2*sigma1_h**2)*\
            np.sum([(L_window[i] - np.e**(-x[0]*delta1)*L_window[i-1])**2 for i in range(1,len(L_window),2)]) - (N-1)/2*np.log(2*np.pi)-\
            (N-1)*np.log(sigma2_h)- 1/(2*sigma2_h**2)*\
            np.sum([(L_window[i] - np.e**(-x[0]*delta2)*L_window[i-1])**2 for i in range(2,len(L_window),2)])
            return -f

        def get_initial_delta(L_window):
            L_day = [L_window[i] - L_window[i-1] for i in range(1,len(L_window),2)]
            L_night = [L_window[i] - L_window[i-1] for i in range(2,len(L_window),2)]
            var1 = np.var(L_day)
            var2 = np.var(L_night)
            t = var1/var2
            delta1 = 1/250 * t/(t+1)
            delta2 = 1/250 - delta1
            return [delta1,delta2,t]

        def solve(x):
            return np.array([x[0]+x[1]-1/250,
                           t - (1-np.exp(-2*theta_L*x[0]))/(1-np.exp(-2*theta_L*x[1]))])

        delta1,delta2,t = get_initial_delta(L_window)

    #     print(MLE([0.2,0.7]))
    #     print(MLE([0.6,0.5]))

        while True:
            #print(theta_old, sigma_old)
            opt = scipy.optimize.minimize(MLE, [theta_old,sigma_old], method='L-BFGS-B',args=(), jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
            theta_L, sigma_L = opt.x
            #print(theta_L,sigma_L)
            #print("theta_L=",theta_L)
            if np.abs(theta_old-theta_L) < 10**-6:
                break
            theta_old = theta_L
            sigma_old = sigma_L
            delta1, delta2 = fsolve(solve,[delta1,delta2])

        return theta_L, sigma_L,delta1,delta2

    L_window = []
    Y_window = Y_100[-79*30:]   #后30天5min数据
    for i in Y_100.groupby(level = 0):
        L_window += [i[1][0],i[1][-1]]    #100天开盘价与收盘价
#     print(L_window)
    L_used_for_Y = L_window[70*2-1:100*2]  #后30天开盘价与收盘价+倒数第31天的收盘价
    #L_used_for_Y = [0] + L_window[70*2:100*2]
    L_tilta = [(L_used_for_Y[2*i]+L_used_for_Y[2*i+1])/2 for i in range(int(len(L_used_for_Y)/2))]  #均值项

    theta_L, sigma_L, delta1, delta2 = estimate_for_L(L_window,theta_old,sigma_old)

    theta_Y, sigma_Y = estimate_for_Y(L_tilta, Y_window, L_used_for_Y,delta1)



    return theta_Y, sigma_Y, theta_L, sigma_L, delta1, delta2






def pairs_selection(n_pairs, stocks_pool,start_date):
    data1 = data.copy()
    start_index = data1[data1["date"] == start_date].index[0] # 今天开盘的第一个index
    data_start = data1.loc[start_index-79*100:start_index-1]
    data_start.set_index(["date","time"],inplace=True)
    pairs = []
    for i in combinations(stock_pool,2):
        test = data_start[[i[0],i[1]]]
        Y_100 = np.log(test[i[0]]/test[i[0]][0]) -  np.log(test[i[1]]/test[i[1]][0])
        #Y_100 = Y[:79*100]
        theta_Y, sigma_Y, theta_L, sigma_L, delta1, delta2 = (estimate_for_parameters(Y_100,9,0.2))
        if theta_Y > 0.1 and theta_L >0:
            L_var = sigma_L/2/theta_L*(1-np.exp(-2*theta_L*(delta1+delta2)))
            Y_var = sigma_Y/2/theta_Y*(1-np.exp(-2*theta_Y*delta1/78))
            pairs.append([i,L_var,Y_var])
            #print("the pair is "+ "\t" + i[0]+ "\t" + "and" + "\t" + i[1] + "\t" + "the     theta_Y     is" +"\t"+  str(theta_Y))

    pairs.sort(key = lambda x:x[1])
    temp = 0
    for i in pairs:
        i.append(temp)
        temp += 1

    pairs.sort(key = lambda x:x[2],reverse = True)
    temp = 0
    for i in pairs:
        i.append(temp)
        temp += 1


    pairs.sort(key = lambda x: x[3]+x[4])

    res = [i[0] for i in pairs]

    if len(res) < n_pairs:
        return res

    else: return res[:n_pairs]
