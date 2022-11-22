#include "include/engine.h"
#include <random>

using std::vector;
using engine::Boid, engine::vec2;

const size_t N_BOIDS = 30;

class Boids {
public:
    Boids(size_t n);
    void update_boids(vec2 attractor_pos);
    vector<Boid>* vec() { return boids_current_; }

private:
    // We have two boids_ vectors; one is current, and the other is used to store new computed values.
    // After the new values are computed, the vectors' roles are swapped.
    vector<Boid> boids1_;
    vector<Boid> boids2_;
    // `current` points to whichever of `boids1_` and `boids2_` is current,
    // and `new` points to whichever one is the target for the next computation.
    vector<Boid>* boids_current_;
    vector<Boid>* boids_new_;
    // switches the `current` and `new` pointers
    void swap_boidvec_roles() { vector<Boid>* tmp = boids_current_; boids_current_ = boids_new_; boids_new_ = tmp; }

    // constants
    static constexpr float COLLISION_AVOIDANCE_RADIUS = 0.1; // @todo enable
    static constexpr float SPEED_CAP = 1.0;
    static constexpr float DT = 0.01; // time step
    // weight factors; modify to increase or decrease priority of different factors
    static constexpr float WEIGHT_SEPARATION = 0.02;
    static constexpr float WEIGHT_COHESION   = 0.5;
    static constexpr float WEIGHT_ALIGNMENT  = 1.0;
    static constexpr float WEIGHT_ATTRACTION = 1.0;
};

Boids::Boids(size_t n_boids) {
    // set up RNG stuff
    std::random_device true_rng;
    std::mt19937 generator(true_rng());
    std::uniform_real_distribution<float> rng(-1.0, 1.0);

    // resize vectors
    boids1_.resize(n_boids);
    boids2_.resize(n_boids);
    // initialize pointers
    boids_current_ = &boids1_;
    boids_new_     = &boids2_;

    // initialize the `current` boids vector
    for (Boid& boid : *boids_current_) {
        boid.pos = vec2(rng(generator), rng(generator));
        boid.vel = vec2(SPEED_CAP, SPEED_CAP);
    }
    // We don't need to initialize the `new` vector, since:
    // 1. it isn't used for anything until we first compute the new boids_ data
    // 2. its values will be overwritten when we first compute the new boids_ data
}

void Boids::update_boids(vec2 attractor_pos) {
    size_t n_boids = boids_current_->size();
    float reciprocal_n_other_boids = 1.0 / (float)(n_boids - 1);

    // Compute sums of boid positions and velocities.
    // These will be used later to compute means.
    vec2 sum_pos(0.0);
    vec2 sum_vel(0.0);
    for (Boid& b : *boids_current_) {
        sum_pos += b.pos;
        sum_vel += b.vel;
    }

    // compute new values
    // @todo this could be done in n(n-1)/2 inner loop iterations instead of ~n^2, by updating both `boid` and
    // `other_boid` in the acc_separation step.
    for (size_t i = 0; i < n_boids; ++i) {
        const Boid& boid = (*boids_current_)[i]; // const because we'll write the new values to `boids_new_`
        // Compute acceleration due to cohesion
        vec2 mean_pos = (sum_pos - boid.pos) * reciprocal_n_other_boids; // mean position of all other boids
        vec2 acc_cohesion = mean_pos - boid.pos;

        // Compute acceleration due to alignment
        vec2 mean_vel = (sum_vel - boid.vel) * reciprocal_n_other_boids; // mean velocity of all other boids
        vec2 acc_alignment = mean_vel - boid.vel;

        // Compute acceleration due to separation
        vec2 acc_separation(0.0);
        // avoid other boids
        for (size_t j = 0; j < boids_current_->size(); ++j) {
            if (i == j) continue; // the boid shouldn't try to separate from itself
            const Boid& other_boid = (*boids_current_)[j];

            vec2 displacement = boid.pos - other_boid.pos;
            float distance = glm::length(displacement);
            if (distance < COLLISION_AVOIDANCE_RADIUS) {
                // accelerate away from the other boid
                // @todo note this isn't the way the reference pseudocode does it; I thought it was weird.
                // disp / (dist^2) has magnitude proportional to (1 / dist). This means we accelerate away
                // faster from boids that are closer to us, but may cause instability due to very large
                // accelerations when very close to a boid.
                acc_separation += displacement / (distance * distance);
            }
        }

        // Compute acceleration due to attraction to object (e.g. the object is bread and the birds want it)
        // @todo not sure why I decided to normalize, but it looks cool and reduces overshooting severity
        vec2 acc_attraction = glm::normalize(attractor_pos - boid.pos);

        // write the new values to `boids_new_`
        Boid& new_boid = (*boids_new_)[i];
        // compute net acceleration
        vec2 net_acc =
            WEIGHT_COHESION   * acc_cohesion +
            WEIGHT_SEPARATION * acc_separation +
            WEIGHT_ALIGNMENT  * acc_alignment +
            WEIGHT_ATTRACTION * acc_attraction;
        // compute velocity
        new_boid.vel = boid.vel + DT * net_acc;
        float speed = length(new_boid.vel);
        if (speed > SPEED_CAP) new_boid.vel *= SPEED_CAP / speed; // enforce speed limit
        // compute position
        new_boid.pos = boid.pos + DT * new_boid.vel;
        // wrap around if hit window border
        // @todo disable this once we have the "follow the tracked object" rule; just let them leave the
        // border, since they'll come back anyway.
        if (new_boid.pos.x < -1.0) new_boid.pos.x += 2;
        else if (new_boid.pos.x > 1.0) new_boid.pos.x -= 2;
        if (new_boid.pos.y < -1.0) new_boid.pos.y += 2;
        else if (new_boid.pos.y > 1.0) new_boid.pos.y -= 2;
    }

    // update pointers to reflect which vector has the current boid data
    swap_boidvec_roles();
}

int main() {
    Boids boids(N_BOIDS);
    vec2 attractor_pos(0.0);

    engine::Engine eng;
    // @todo engine needs a way to initialize boids before calling run(). The following is a placeholder hack.
    bool first_iter = true;
    eng.run([&](){
        if (first_iter) {
            eng.update_boids(*boids.vec());
            first_iter = false;
        }
    });
}