import numpy as np

def create_ec_pdb(filename, n_side=4, spacing=1.2):
    # Base coordinates for 1 EC molecule (rough estimate)
    # Atoms: O00, C01, O02, C03, C04, O05, H06, H07, H08, H09
    base_coords = np.array([
        [ 0.000,  0.120,  0.000], # O00
        [ 0.000, -0.000,  0.000], # C01
        [ 0.110, -0.070,  0.000], # O02
        [ 0.070, -0.210,  0.000], # C03
        [-0.070, -0.210,  0.000], # C04
        [-0.110, -0.070,  0.000], # O05
        [ 0.120, -0.250,  0.080], # H06
        [ 0.120, -0.250, -0.080], # H07
        [-0.120, -0.250,  0.080], # H08
        [-0.120, -0.250, -0.080]  # H09
    ])
    
    atom_names = ["O00", "C01", "O02", "C03", "C04", "O05", "H06", "H07", "H08", "H09"]
    elements = ["O", "C", "O", "C", "C", "O", "H", "H", "H", "H"]
    
    with open(filename, 'w') as f:
        f.write("REMARK   EC box\n")
        f.write(f"CRYST1{n_side*spacing*10:9.3f}{n_side*spacing*10:9.3f}{n_side*spacing*10:9.3f}  90.00  90.00  90.00 P 1           1\n")
        atom_count = 1
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    offset = np.array([i, j, k]) * spacing
                    for idx, (name, elem) in enumerate(zip(atom_names, elements)):
                        pos = (base_coords[idx] + offset) * 10.0 # to Angstrom
                        res_id = (atom_count - 1) // 10 + 1
                        f.write(f"ATOM  {atom_count:5d} {name:4s} ECA A{res_id:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {elem}\n")
                        atom_count += 1
        f.write("END\n")

if __name__ == "__main__":
    create_ec_pdb("ec_init.pdb")
