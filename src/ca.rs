extern crate rand;
extern crate rayon;
extern crate bytecount;
extern crate nalgebra_glm as glm;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<u8>,
    visible_cells: Vec<usize>,
    sectors: Vec<Vec<usize>>,
    width: usize,
    height: usize,
    length: usize,
    min_surv: u8,
    max_surv: u8,
    min_birth: u8,
    max_birth: u8,
    max_age: u8
}

impl CellA {
    pub fn new ( width: usize, height: usize, length: usize, min_surv: u8, max_surv: u8, min_birth: u8, max_birth: u8 ) -> Self {
        let cells = vec![0; width * height * length];
        let max_age = 1;

        Self {
            cells,
            visible_cells: Vec::new(),
            sectors: Vec::new(),
            width,
            height,
            length,
            min_surv,
            max_surv,
            min_birth,
            max_birth,
            max_age,
        }
    }

    pub fn update_visible ( &mut self ) {
        let start = std::time::Instant::now();
        self.visible_cells = self.cells
            .par_iter()
            .enumerate()
            .filter_map(|e| {
                let idx = e.0;
                if (idx > self.width * self.height + self.length) && (idx < (self.width * self.height* self.length) - (self.width * self.height) - self.width - 1) {
                    let neighbors = [
                        self.cells[idx + (self.width * self.height) + self.width + 1],
                        self.cells[idx + (self.width * self.height) + self.width    ],
                        self.cells[idx + (self.width * self.height) + self.width - 1],
                        self.cells[idx + (self.width * self.height)              + 1],
                        self.cells[idx + (self.width * self.height)                 ],
                        self.cells[idx + (self.width * self.height)              - 1],
                        self.cells[idx + (self.width * self.height) - self.width + 1],
                        self.cells[idx + (self.width * self.height) - self.width    ],
                        self.cells[idx + (self.width * self.height) - self.width - 1],
                        self.cells[idx                              + self.width + 1],
                        self.cells[idx                              + self.width    ],
                        self.cells[idx                              + self.width - 1],
                        self.cells[idx                                           + 1],
                        self.cells[idx                                           - 1],
                        self.cells[idx                              - self.width + 1],
                        self.cells[idx                              - self.width    ],
                        self.cells[idx                              - self.width - 1],
                        self.cells[idx - (self.width * self.height) + self.width + 1],
                        self.cells[idx - (self.width * self.height) + self.width    ],
                        self.cells[idx - (self.width * self.height) + self.width - 1],
                        self.cells[idx - (self.width * self.height)              + 1],
                        self.cells[idx - (self.width * self.height)                 ],
                        self.cells[idx - (self.width * self.height)              - 1],
                        self.cells[idx - (self.width * self.height) - self.width + 1],
                        self.cells[idx - (self.width * self.height) - self.width    ],
                        self.cells[idx - (self.width * self.height) - self.width - 1]
                    ];

                    let count: u8 = neighbors.iter().sum();
                    if count == 26 {
                        None
                    } else {
                        Some(idx)
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        println!("Updating visible cells took: {} s", super::utils::get_elapsed(start));
    }

    pub fn recalculate_sectors ( &mut self, cube_offsets: &[[f32; 3]] ) {
        let start = std::time::Instant::now();
        // sectors are also organized in z * (width * height) + y * width + x, corresponding to camera position
        // sectors are 32x32x32
        let sector_size = 32;
        // todo: did I get these right?
        let num_sectors_x = self.width / sector_size;
        let num_sectors_y = self.height / sector_size;
        let num_sectors_z = self.length / sector_size;
        let total_sectors = num_sectors_x * num_sectors_y * num_sectors_z;

        self.sectors = (0 .. total_sectors)
            .into_par_iter()
            .map(|sec_idx| {
                // get true center x y z
                let sec_z = sec_idx / (num_sectors_z * num_sectors_y);
                let sec_y = (sec_idx % (num_sectors_y * num_sectors_x)) / num_sectors_y;
                let sec_x = sec_idx % num_sectors_x;

                let center_x = (sec_x * sector_size) as f32;
                let center_y = (sec_y * sector_size) as f32;
                let center_z = (sec_z * sector_size) as f32;

                // filter visible to only include near
                self.visible_cells.iter()
                    .filter_map(|&idx| {
                        let distance = glm::distance(
                            &glm::vec3(center_x as f32, center_y as f32, center_z as f32),
                            &glm::vec3(cube_offsets[idx][0], cube_offsets[idx][1], cube_offsets[idx][2])
                        );

                        // arbitrary threshold
                        if distance < 60.0 {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<Vec<usize>>>();
        println!("Calculating sectors took: {} s", super::utils::get_elapsed(start));
    }

    pub fn get_near_and_visible ( &self, camera_position: &glm::Vec3 ) -> Vec<usize> {
        // todo: no magic numbers
        let sector_size = 32;
        let num_sectors_x = (self.width  / sector_size) as f32;
        let num_sectors_y = (self.height / sector_size) as f32;
        let num_sectors_z = (self.length / sector_size) as f32;
        let scaled_x = (camera_position.x / (self.width  as f32) * num_sectors_x) as usize;
        let scaled_y = (camera_position.y / (self.height as f32) * num_sectors_y) as usize;
        let scaled_z = (camera_position.z / (self.length as f32) * num_sectors_z) as usize;
        println!("Scaled xyz: {}, {}, {}", scaled_x, scaled_y, scaled_z);

        let sector_idx = scaled_z * (num_sectors_x as usize * num_sectors_y as usize) + scaled_y * num_sectors_x as usize + scaled_x;

        self.sectors[sector_idx as usize].clone()
    }

    // pub fn set_xyz ( &mut self, x: usize, y: usize, z: usize, new_state: u8 ) {
    //     let idx = (z * self.width * self.height) + (y * self.width) + x;
    //     self.cells[idx] = new_state;
    // }

    pub fn randomize ( &mut self ) {
        let cells = (0 .. self.width * self.height * self.length)
            .map(|_| {
                // if x < self.width * self.height {
                    if rand::random() {
                        1
                    } else {
                        0
                    }
                // } else {
                //     0
                // }
            })
            .collect();

        self.cells = cells;
    }

    pub fn next_gen ( &mut self ) {
        let new_cells = (0 .. self.width * self.height * self.length)
            .into_par_iter()
            .map(|idx| {
                if (idx > self.width * self.height + self.width) && (idx < (self.width * self.height * self.length) - (self.width * self.height) - self.width - 1) {
                    let cur_state = self.cells[idx];
                    let neighbors = [
                        self.cells[idx + (self.width * self.height) + self.width + 1],
                        self.cells[idx + (self.width * self.height) + self.width    ],
                        self.cells[idx + (self.width * self.height) + self.width - 1],
                        self.cells[idx + (self.width * self.height)              + 1],
                        self.cells[idx + (self.width * self.height)                 ],
                        self.cells[idx + (self.width * self.height)              - 1],
                        self.cells[idx + (self.width * self.height) - self.width + 1],
                        self.cells[idx + (self.width * self.height) - self.width    ],
                        self.cells[idx + (self.width * self.height) - self.width - 1],
                        self.cells[idx                              + self.width + 1],
                        self.cells[idx                              + self.width    ],
                        self.cells[idx                              + self.width - 1],
                        self.cells[idx                                           + 1],
                        self.cells[idx                                           - 1],
                        self.cells[idx                              - self.width + 1],
                        self.cells[idx                              - self.width    ],
                        self.cells[idx                              - self.width - 1],
                        self.cells[idx - (self.width * self.height) + self.width + 1],
                        self.cells[idx - (self.width * self.height) + self.width    ],
                        self.cells[idx - (self.width * self.height) + self.width - 1],
                        self.cells[idx - (self.width * self.height)              + 1],
                        self.cells[idx - (self.width * self.height)                 ],
                        self.cells[idx - (self.width * self.height)              - 1],
                        self.cells[idx - (self.width * self.height) - self.width + 1],
                        self.cells[idx - (self.width * self.height) - self.width    ],
                        self.cells[idx - (self.width * self.height) - self.width - 1]
                    ];

                    let count: u8 = neighbors.iter().sum();

                    if cur_state > 0 {
                        if count >= self.min_surv && count <= self.max_surv {
                            let new_state = cur_state + 1;
                            if new_state > self.max_age {
                                self.max_age
                            } else {
                                new_state
                            }
                        } else {
                            0
                        }
                    } else if count >= self.min_birth && count <= self.max_birth {
                        1
                    } else {
                        0
                    }

                    // q-states
                    // if cur_state > self.max_age / 2 {
                    //     if self.min_surv <= count && count <= self.max_surv {
                    //         if cur_state < self.max_age {
                    //             cur_state + 1
                    //         } else {
                    //             self.max_age
                    //         }
                    //     } else {
                    //         cur_state - 1
                    //     }
                    // } else {
                    //     if self.min_birth <= count && count <= self.max_birth {
                    //         if cur_state < self.max_age {
                    //             cur_state + 1
                    //         } else {
                    //             self.max_age
                    //         }
                    //     } else {
                    //         if cur_state > 0 {
                    //             cur_state - 1
                    //         } else {
                    //             0
                    //         }
                    //     }
                    // }
                } else {
                    0
                }
            })
            .collect();

        self.cells = new_cells;
    }
}
