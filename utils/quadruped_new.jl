using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using Random
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff
using GeometryBasics: HyperRectangle, Vec
using MeshCat: setobject!, MeshPhongMaterial, Translation, LinearMap
using Rotations: RotZ
using Colors

const URDFPATH = joinpath(@__DIR__,"a1","urdf","a1.urdf")

function init_quadruped()
    a1 = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false) 
    return a1
end

struct UnitreeA1{C}
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function UnitreeA1(mech::Mechanism)
        nx = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(nx)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end
function UnitreeA1()
    UnitreeA1(init_quadruped())
end

state_dim(model::UnitreeA1) = num_positions(model.mech) + num_velocities(model.mech)
control_dim(model::UnitreeA1) = 12 

function dynamics(model::UnitreeA1, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1,T2} 
    T = promote_type(T1,T2)
    state = model.statecache[T]
    res = model.dyncache[T]

    # Convert from state ordering to the ordering of the mechanism
    copyto!(state, x)
    τ = [zeros(6); u]
    dynamics!(res, state, τ)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

function initial_state(model::UnitreeA1)
    state = model.statecache[Float64]
    a1 = model.mech
    zero!(state)
    leg = ("FR","FL","RR","RL")
    for i = 1:4
        # s = isodd(i) ? 1 : -1
        # f = i < 3 ? 1 : -1
        set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), deg2rad(00))
        set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), deg2rad(50))
        set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), deg2rad(-100))
    end
    floating_joint = findjoint(a1, "base_to_world")
    trunk_height = get_foot_position(model, configuration(state))[3]
    set_configuration!(state, floating_joint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -trunk_height])

    return [configuration(state); velocity(state)]
end

function goal_state(model::UnitreeA1)
    state = model.statecache[Float64]
    a1 = model.mech
    zero!(state)
    floating_joint = findjoint(a1, "base_to_world")
    trunk_height = get_foot_position(model, configuration(state))[3]
    set_configuration!(state, floating_joint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -trunk_height])

    return [configuration(state); velocity(state)]
end

# function goal_state(model::UnitreeA1)
#     state = model.statecache[Float64]
#     a1 = model.mech
#     zero!(state)
#     leg = ("FR","FL","RR","RL")
#     for i = 1:2
#         s = isodd(i) ? 1 : -1
#         f = i < 3 ? 1 : -1
#         # set FR, FL
#         set_configuration!(state, findjoint(a1, leg[i] * "_hip_joint"), -0.009836778663223645s)
#         set_configuration!(state, findjoint(a1, leg[i] * "_thigh_joint"), 0.016964453132483102)
#         set_configuration!(state, findjoint(a1, leg[i] * "_calf_joint"), 0.004758019542372259)

#         # set RR, RL 
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_hip_joint"), -0.0007997782745616991s)
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_thigh_joint"), 0.1272482316068922)
#         set_configuration!(state, findjoint(a1, leg[i + 2] * "_calf_joint"), 0.23113053507838607)
#     end
#     floating_joint = findjoint(a1, "base_to_world")
#     trunk_height = get_foot_position(model, configuration(state))[3]
#     # TODO: rotate trunk to appropriate pos
#     set_configuration!(state, floating_joint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -trunk_height])

#     return [configuration(state); velocity(state)]
# end

##########################
##    GET FUNCTIONS     ## 
##########################

function get_trunk_position(model::UnitreeA1, q)
    mech = model.mech 
    T = eltype(q)
    state = MechanismState{T}(mech)
    set_configuration!(state, q)

    trunk = findbody(mech, "base")
    tf_world = transform_to_root(state, default_frame(trunk))
    # world = findbody(mech, "world") 
    # tf_world = relative_transform(state, default_frame(world), default_frame(trunk))
    return translation(tf_world)
end

function get_foot_position(model::UnitreeA1, q, foot="RR")
    mech = model.mech 
    T = eltype(q)
    state = MechanismState{T}(mech)
    set_configuration!(state, q)

    foot_body = findbody(mech, foot * "_foot")
    tf_world = transform_to_root(state, default_frame(foot_body))
    # world = findbody(mech, "world") 
    # tf_world = relative_transform(state, default_frame(world), default_frame(foot_body))
    return translation(tf_world)
end

function get_trunk_velocity(model::UnitreeA1, x)
    mech = model.mech 
    T = eltype(x)
    state = MechanismState{T}(mech)
    copyto!(state, x)
    trunk = findbody(mech, "trunk")
    twist = twist_wrt_world(state, trunk)
    v = linear(twist)
    return v
end

##########################
## VISUALIZER FUNCTIONS ## 
##########################

function initialize_visualizer(a1::UnitreeA1)
    vis = Visualizer()
    delete!(vis)
    cd(joinpath(@__DIR__,"a1","urdf"))
    
    # Create robot visualization
    mvis = MechanismVisualizer(a1.mech, URDFVisuals(URDFPATH), vis)
    
    # Add block to the same visualizer
    add_block(vis)  # Using default parameters
    
    cd(@__DIR__)
    return mvis
end

function add_block(vis::Visualizer; 
    name::String="block",
    position=Translation(0.4, -0.3, 0.),  # Default position: 0.5m in front, centered
    size=Vec(1.0, 0.6, 0.3),              # 10cm cube
    color=RGB(0.8, 0.2, 0.2)              # Red color
)
    # Create and configure block geometry
    block = HyperRectangle(Vec(0.0, 0.0, 0.0), size)
    setobject!(vis[name], block, MeshPhongMaterial(color=color))
    
    # Apply position and orientation transformations
    settransform!(vis[name], position ∘ LinearMap(RotZ(0.0)))
end

##########################
##        UTILS         ## 
##########################

function create_idx(nx,nu,N)
    # This function creates some useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]    
    
    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N-1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1 : nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx + 1):(nx + nu)) for i = 1:(N - 1)]
    
    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx-1) .+ (1 : nx-1) for i = 1:(N - 1)]
    nc = (N - 1) * (nx-1) # (N-1)*nx 
    
    return (nx=nx,nu=nu,N=N,nz=nz,nc=nc,x= x,u = u,c = c)
end