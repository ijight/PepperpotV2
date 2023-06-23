import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import os
import csv
import json
import time

class Particle:
    def __init__(self, Nseed, iq, dt, dW, x, x_prime, y, y_prime):
        self.Nseed = Nseed
        self.iq = iq
        self.dt = dt
        self.dW = dW
        self.x = x
        self.x_prime = x_prime
        self.y = y
        self.y_prime = y_prime
        self.z = 0

    def __repr__(self):
        return f"Particle(Nseed={self.Nseed}, iq={self.iq}, dt={self.dt}, dW={self.dW}, x={self.x}, x'={self.x_prime}, y={self.y}, y'={self.y_prime})"

    def propagate(self, distance):
        import math
        self.x += distance * math.tan(self.x_prime / 1000)
        self.y += distance * math.tan(self.y_prime / 1000)

class Hole:
    def __init__(self, center_x, center_y, center_z, diameter):
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.diameter = diameter

    def __repr__(self):
        return f"Hole(center_x={self.center_x}, center_y={self.center_y}, center_z={self.center_z}, diameter={self.diameter})"
    
class Grid:
    def __init__(self, size_x, size_y, size_z, hole_diameter, separation):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.hole_diameter = hole_diameter
        self.separation = separation
        self.holes = self.create_3d_grid()

    def create_3d_grid(self):
        holes = []
        center_x_offset = ((self.size_x - 1) * self.separation) / 2
        center_y_offset = ((self.size_y - 1) * self.separation) / 2
        center_z_offset = ((self.size_z - 1) * self.separation) / 2
        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    hole = Hole(center_x=x*self.separation - center_x_offset, 
                                 center_y=y*self.separation - center_y_offset, 
                                 center_z=z*self.separation - center_z_offset, 
                                 diameter=self.hole_diameter)
                    holes.append(hole)
        return holes

    def is_point_in_a_hole(self, point):
        x1, y1, z1 = point
        hole_radius = self.hole_diameter / 2

        x_index = round(x1 / self.separation)
        y_index = round(y1 / self.separation)
        z_index = round(z1 / self.separation)

        x_center = x_index * self.separation
        y_center = y_index * self.separation
        z_center = z_index * self.separation

        return (x_center - hole_radius <= x1 <= x_center + hole_radius and
                y_center - hole_radius <= y1 <= y_center + hole_radius and
                z_center - hole_radius <= z1 <= z_center + hole_radius)

def connect_to_database():
    return mysql.connector.connect(
        host='srv395.hstgr.io',
        port='3306',
        user='u641848469_iank',
        password='VPCBwkO68!!nCOXa2Yv4',
        database='u641848469_argonneimages',
        connect_timeout=10000000
)

def reconnect_to_database():
    global cnx
    while True:
        try:
            cnx = connect_to_database()
            print("Reconnected to the MySQL server.")
            break
        except mysql.connector.Error as err:
            print(f"Failed to reconnect to the MySQL server: {err}")

data_folder = "E://data/"
cnx = connect_to_database()

for filename in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, filename)
    if os.path.isdir(folder_path):
        coord_file = os.path.join(folder_path, "coord.out")

        cursor = cnx.cursor()
        sql_check_filename = "SELECT COUNT(*) FROM imagesndata WHERE paramsname = %s"
        cursor.execute(sql_check_filename, (filename,))
        result = cursor.fetchone()
        if result is not None and result[0] > 0:
            print(f"Skipping file {filename}. Data already stored.")
            cursor.close()
            continue
        cursor.close()

        if os.path.isfile(coord_file):
            data = []

            size_x, size_y, size_z = 40, 40, 1
            hole_diameter = 0.01
            separation = 0.3
            grid = Grid(size_x, size_y, size_z, hole_diameter, separation)

            particles_in_holes = []
            for particle in data:
                point = (particle.x, particle.y, particle.z)
                if grid.is_point_in_a_hole(point):
                    particles_in_holes.append(particle)

            for particle in particles_in_holes:
                particle.propagate(10)

            if particles_in_holes:
                x_in_holes = [particle.x for particle in particles_in_holes]
                y_in_holes = [particle.y for particle in particles_in_holes]

            bins_hist = 200
            bins_grid = size_x

            histo_width = (size_x * separation) / 2
            edges_hist = np.linspace(-histo_width, histo_width, bins_hist + 1)

            H, _, _ = np.histogram2d(x_in_holes, y_in_holes, bins=[edges_hist, edges_hist])

            fig, ax = plt.subplots(figsize=(8, 8))

            pcm = ax.pcolormesh(edges_hist, edges_hist, H.T, cmap='inferno')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            plt.axis('square')

            image_path = f"images/{filename}.png"

            plt.savefig(image_path, dpi=64, bbox_inches='tight', pad_inches=0)

            plt.close(fig)

            csv_data = []

            csv_data_str = json.dumps(csv_data)

            beam_file = os.path.join(folder_path, "beam.out")
            beam_data = b""
            with open(beam_file, "rb") as file:
                beam_data = file.read()

            track_file = os.path.join(folder_path, "track.dat")
            track_data = b""
            with open(track_file, "rb") as file:
                track_data = file.read()

            log_file = os.path.join(folder_path, "log.out")
            log_data = b""
            with open(log_file, "rb") as file:
                log_data = file.read()

            sql = "INSERT INTO imagesndata (paramsname, image, data, beam, track, log) VALUES (%s, %s, %s, %s, %s, %s)"
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            try:
                cursor = cnx.cursor()
                cursor.execute(sql, (filename, image_data, csv_data_str, beam_data, track_data, log_data))
                cnx.commit()
                cursor.close()
                print(f"Data saved to database for file: {filename}")
            except mysql.connector.Error as err:
                print(f"MySQL Error occurred for file {filename}: {str(err)}")
                if err.errno == 2013 or err.errno == 2055:
                    reconnect_to_database()

cursor.close()
cnx.close()
