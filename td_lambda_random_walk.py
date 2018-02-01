import numpy as np
#np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
import math

class RandomWalk:

    def __init__(self):
        self.nodes = ["A", "B", "C", "D", "E", "F", "G"]
        self.position = 3 
        self.node = self.nodes[self.position]
	self.node_vals = [[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,1,1,1,1]]
        self.terminated = False

    def move(self):
        if not self.terminated:
            direction = np.random.choice([0,1])
            if direction == 0:
                new_position = self.position - 1
            elif direction == 1:
                new_position = self.position + 1
            self.position = new_position
            self.node = self.nodes[self.position]
            if (self.node == "A") or (self.node == "G"):
                self.terminated = True

    def move_sequence(self):
        sequence = []
        letter_seq = []
        while True:
            sequence.append(self.node_vals[self.position])
	    letter_seq.append(self.nodes[self.position])
	    self.move()
	    if self.terminated == True:
                sequence.append(self.node_vals[self.position])
                letter_seq.append(self.nodes[self.position])
                break
	return [sequence,letter_seq]
              
# Generate 100 copies by 10 episodes sequence data for use in the analysis
N_episodes = 100
all_episodes = []
for episode in range(N_episodes):
    for one_eps in range(10):
        eps = []
        c = RandomWalk()
        new_seq,lseq = c.move_sequence()
        eps.append(new_seq)
    all_episodes.append(eps)

# Calculate the weight of a given episode
def calc_pt(seq_episode,w):
    if seq_episode   == [1,1,1,1,1]:
       Pt = 1
    elif seq_episode == [0,0,0,0,0]:
       Pt = 0
    else:
       Pt = np.dot(np.transpose(w),seq_episode)
    return Pt 

# Calculate the weights for a single episode
def calc_weights(eps_seq,nlambda,alpha,init_w):
    delta_w = 0.5*np.zeros(5)
    for t in range(len(eps_seq)-1):
        Pt = calc_pt(eps_seq[t],init_w)
        Pt_plus1 = calc_pt(eps_seq[t+1],init_w)
        lambda_sum = 0.0
        for k in range(t+1):
            lambda_sum = lambda_sum + nlambda**(t-k) * np.array(eps_seq[k])
        update = alpha*(Pt_plus1-Pt)*lambda_sum
        delta_w = delta_w + update
    return delta_w

Z = np.array([1./6,1./3,1./2,2./3,5./6])

# Function to run experiments for Figures 3 and 5 
def run_Figure3_experiment(all_episodes,nlambda,alpha):
    full_sum = []
    for ten_seq in all_episodes:
        init_w = 0.01*np.ones(5)
        while True:
            sum_w = np.zeros(5)
            for one_seq in ten_seq:
                weights = calc_weights(one_seq,nlambda,alpha,init_w)
	        sum_w = sum_w + weights
            intermed_w = init_w + sum_w
            converged = intermed_w - init_w
            if sum(abs(converged)) < 0.0000001:
	        full_sum.append(intermed_w)
                break
            else:
                init_w = intermed_w
    ave_w = sum(full_sum)/len(full_sum)
    rmse = np.sqrt( np.mean( (ave_w-Z)**2 ) )
    return  rmse

# Function to run experiments for Figure 4
def run_Figure4_experiment(all_episodes,nlambda,alpha):
    full_sum = []
    for ten_seq in all_episodes:
        init_w = 0.5*np.ones(5)
        for one_seq in ten_seq:
            weights = calc_weights(one_seq,nlambda,alpha,init_w)
            init_w = init_w + weights
        full_sum.append(init_w)
    #ave_w = sum(full_sum)/100.0
    #rmse = np.sqrt(np.mean((ave_w-Z)**2))
    rmse = np.sqrt(np.mean((full_sum-Z)**2))
    return  rmse


nlambdas = np.arange(0.0,1.1,0.1)
RMSEs = []
alpha = 0.01
alphas = [0.6,0.6,0.6,0.6,0.6,0.6,0.45,0.4,0.2,0.15,0.05]
niter = 0

for nlambda in nlambdas:
    rmse = run_Figure3_experiment(all_episodes,nlambda,alpha)
    #rmse = run_Figure3_experiment(all_episodes,nlambda,alphas[niter])
    RMSEs.append(rmse)
    niter += 1

plt.plot(nlambdas,RMSEs, marker = '.', ms=14, color='magenta')
plt.plot(nlambdas,RMSEs, 'r-', ms=8, color='magenta')
plt.xlabel(r'$\mathrm{\lambda}$', family='sans-serif',size='18',stretch='ultra-condensed',color='k')
plt.ylabel('RMSE using best $\mathrm{\alpha}$')
#plt.savefig('Figure3.pdf')
#plt.savefig('Figure5.pdf')

nlambdas = [0.,0.3,0.8,1.0]
alphas = np.arange(0.0,0.7,0.05)
colors = ['black','red','magenta','green']
labels = [r'$\mathrm{\lambda}$ = 0', r'$\mathrm{\lambda}$ = 0.3', r'$\mathrm{\lambda}$ = 0.8',r'$\mathrm{\lambda}$ = 1 (Widrow-Hoff)']
itr = 0
for nlambda in nlambdas:
    RMSEs = []
    for alpha in alphas:     
        rmse = run_Figure4_experiment(all_episodes,nlambda,alpha)
        RMSEs.append(rmse)
    plt.plot(alphas,RMSEs, marker = '.', ms=14) #, color = colors[itr])
    plt.plot(alphas,RMSEs, ls = '-', ms=8)# ,  color = colors[itr])
    plt.text(alphas[-1]+0.01,RMSEs[-1]+0.01,labels[itr], {'color' : 'k', 'fontsize' : 10},horizontalalignment = 'left',verticalalignment = 'center',rotation = 0,clip_on = False)    
    itr += 1
plt.ylim([0.15,0.8])
plt.xlim([-0.1,1.2])
plt.xlabel(r'$\mathrm{\alpha}$', family='sans-serif',size='14',stretch='ultra-condensed',color='k')
plt.ylabel('RMSE')
plt.title('Figure 4', family='sans-serif',size='18',stretch='ultra-condensed',color='r')
plt.savefig('Figure4.pdf')

