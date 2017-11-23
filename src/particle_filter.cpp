/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
  num_particles = 100;
  // Resize all vectors by actual size
  particles.resize(num_particles);
  weights.resize(num_particles);

  // Define normal distributions for noise
  default_random_engine gen;

  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  // Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  for (int i = 0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].x = x;
    particles[i].y = y;
    particles[i].theta = theta;
    particles[i].weight = 1.0;

    // Add random Gaussian noise to each particle.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Define normal distributions for noise
  default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // Add measurements to each particle and add random Gaussian noise.
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) >= THRESHOLD) {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }
    else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // Changes of theta is too small to apply it
    }

    // Add random Gaussian noise to each particle.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
  for (int i = 0; i < observations.size(); i++) {
    double min_dist = numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_dist) {
        min_dist = distance;
        // Replace ID to has minimal distance
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution

  for (int i = 0; i < num_particles; i++) {
    // Getting particle coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // Reset weight
    particles[i].weight = 1.0;

    // Collect map landmarks within sensor range distance to particles

    // Vector for predicted landmarks within sensor range
    vector<LandmarkObs> predictions;

    // Go through map landmarks
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      int map_id = map_landmarks.landmark_list[j].id_i;
      float map_x = map_landmarks.landmark_list[j].x_f;
      float map_y = map_landmarks.landmark_list[j].y_f;
      double distance = dist(p_x, p_y, map_x, map_y);

      if (distance < sensor_range) {
        predictions.push_back(LandmarkObs{ map_id, map_x, map_y });
      }
    }

    // Transform observation coordinates from car coordinate system to map coordinate system
    vector<LandmarkObs> map_observations;
    for (int j = 0; j < observations.size(); j++) {
      double map_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      double map_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
      // Set id to 0, will be set up correctly in data association step
      map_observations.push_back(LandmarkObs{ 0, map_x, map_y });
    }

    // Make a data association
    dataAssociation(predictions, map_observations);

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

    // Evaluate weight
    for (int j = 0; j < map_observations.size(); j++) {
      double x_obs = map_observations[j].x;
      double y_obs = map_observations[j].y;
      double x_pred;
      double y_pred;
      for (int k = 0; k < predictions.size(); k++) {
        if (map_observations[j].id == predictions[k].id) {
          x_pred = predictions[k].x;
          y_pred = predictions[k].y;
        }
      }
      double x_exponent = pow(x_obs - x_pred, 2) / (2 * pow(sig_x, 2));
      double y_exponent = pow(y_obs - y_pred, 2) / (2 * pow(sig_y, 2));
      double exponent = x_exponent + y_exponent;
      double weight = gauss_norm * exp(-exponent);
      particles[i].weight *= weight;
    }
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.

  vector<Particle> resampled(num_particles);

  random_device rd;
  default_random_engine gen(rd());
  // Distribution according to weights
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  // Resample the particles according to their weights
  for (int i = 0; i < num_particles; ++i) {
    int idx = dist(gen);
    resampled[i] = particles[idx];
  }

  // Replace particles with the resampled particles
  particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
