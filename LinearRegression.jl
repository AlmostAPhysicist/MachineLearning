# Non Machine Learning Approach
using CSV, GLM, Plots, TypedTables

data = CSV.File("housePricedata.csv")

X = data.size

Y = round.(Int, data.price / 1000)

t = Table(x=X, y=Y)

gr(size = (600,600))

p = scatter(X, Y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (thousands of dollars)",
    title = "Housing Prices in Portland",
    # legend = false,
    color = :red
)

# Using the GLM package

ols = lm(@formula(y ~ x), t)

plot!(X, predict(ols), color=:green, linewidth=3, label="Analytical Solution")

#########################################
# Machine Learning Approach
#########################################

epochs = 0 #just means the count of iterations completed while optimising and minimising the cost function

p = scatter(X, Y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (thousands of dollars)",
    title = "Housing Prices in Portland (epochs = $epochs)",
    # legend = false,
    color = :red
)

#Parameters
θ_0 = 0 #bias (actually just the weight of x=0)
θ_1 = 0 #weight

# define the linear regression model

h(x) = θ_0 .+ θ_1*x #This is what we hypothesis the function to look like

# The heart is the Cost function...
# For this model, a function similar to the varience of the data about the predicted value can be helpful
# A Function with low varience about the prediction would have a line of better function
# We want to tweek our parameter set θ such that the value of the cost function minimises.

N = length(X)
ypred = ŷ = h(X)
J(X, Y) = (1/(2N)) * sum((ŷ - Y).^2)

J_history = []

push!(J_history, J(X,Y))

#How to minimise the cost function

#Gradient Descent Algorithm 

#To minimise the cost function, you need dJ/dt to be zero.
# i.e. you need to find the root of the differential of the Cost function
# like newton's method of finding a root (x_n+1 = x_n - f(x)/f'(x)) basically, setting the descent rate to be determined by the gradient to fall into the pit of a zero
# we define a very loosely similar expression (x_n+1 = x_n - α f(x)) where alpha is controllable (f(x) here is the derivative of the function since we want ITs root)
# we do this for partial derivative wrt both the variables since we like to control them individually

pd_θ_0(X,Y) = (1/(N)) * sum((ŷ - Y))
pd_θ_1(X,Y) = (1/(N)) * sum((ŷ - Y).*X)

α_0 = 0.09 #This regression rate or the LEARNING rate is for the bias
α_1 = 8.0e-8 # Learning rate for the slope


# Using the Algorithm to Optimise Iteratively (mannual iteration)
θ_0_temp = pd_θ_0(X,Y)
θ_1_temp = pd_θ_1(X,Y)

# adjusting the parameters based off of partial derivatives
θ_0 -= α_0 * θ_0_temp
θ_1 -= α_1 * θ_1_temp


ŷ = h(X)
push!(J_history, J(X, Y))
#The value of the cost function dropped from 65k to 21k! Huge improvement!

epochs += 1

plot!(X, ŷ, color = :blue, alpha = 0.5,
    title = "House Prices in Portland (epochs = $epochs)"
)

# using the gradient Descent Algorithm (with a loop)

for i in 1:100
    θ_0_temp = pd_θ_0(X,Y)
    θ_1_temp = pd_θ_1(X,Y)

    # adjusting the parameters based off of partial derivatives
    θ_0 -= α_0 * θ_0_temp
    θ_1 -= α_1 * θ_1_temp


    ŷ = h(X)
    push!(J_history, J(X, Y))
    #The value of the cost function dropped from 65k to 21k! Huge improvement!

    epochs += 1

    plot!(X, ŷ, color = :blue, alpha = 0.5, label=nothing,
        title = "House Prices in Portland (epochs = $epochs)"
    )
    display(p)

    yield()
    sleep(0.1)
end


display(plot)