from Sarsa import main as Sarsa_main
from QLearning import main as Qlearning_main
from openAIgym import main as openAIgym_main
from sys import argv

def main():
	if '--sarsa' in argv:
		Sarsa_main()
	elif '--qlearning' in argv:
		Qlearning_main()
	elif '--gym' in argv:
		openAIgym_main()

if __name__ == '__main__':
	main()