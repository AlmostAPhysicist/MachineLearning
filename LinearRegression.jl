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
    legend = false,
    color = :red
)

# Using the GLM package

ols = lm(@formula(y ~ x), t)

plot!(X, predict(ols), color=:green, linewidth=3)

#########################################
# Machine Learning Approach
#########################################

epochs = 0

p = scatter(X, Y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (thousands of dollars)",
    title = "Housing Prices in Portland (epochs = $epochs)",
    legend = false,
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
J(θ_0, θ_1) = (1/(2N)) * sum((ŷ - Y).^2)

J_history = []

push!(J_history, J(θ_0, θ_1))
