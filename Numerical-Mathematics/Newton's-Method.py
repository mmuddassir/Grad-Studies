import math

def newton(f, f_prime, x, nmax, epsilon, delta):
    # fx <- f(x)
    fx = f(x)
    
    # output 0, x, fx
    print(f"{0:^2} | {x:^45} | {fx:^45} |")
    
    # for n = 1 to nmax do
    for n in range(1, nmax + 1):
        # fp <- f'(x)
        fp = f_prime(x)
        
        # if |fp| < delta then
        if abs(fp) < delta:
            print("Output: small derivative")
            return
            
        # d <- fx/fp
        d = fx / fp
        # x <- x - d
        x = x - d
        # fx <- f(x)
        fx = f(x)
        
        # output n, x, fx
        print(f"{n:^2} | {x:^45} | {fx:^45} | ")
        
        # if |d| < epsilon then
        if abs(d) < epsilon:
            print("Output: convergence\n")
            return
            
    print("Output: Max attempts reached without convergence\n")

# -------------------------------------------------------------------
# functions and derivatives go here

def f(t):
    return 2*math.sin(t)

def f_prime(t):

    return 2*math.cos(t)



# -------------------------------------------------------------------
# Run Parameters and any changes to display stuff should be made here
NMAX = 100
EPSILON = 1e-6
DELTA = 1e-6


print("-"*100)
print(f"{'n':^2} | {'x':^45} | {'f(x)':^45} |")
print("-"*100)
newton(f, f_prime, x=7, nmax=NMAX, epsilon=EPSILON, delta=DELTA)
