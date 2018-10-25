extern crate rand;
extern crate rayon;
extern crate bytecount;
extern crate nalgebra_glm as glm;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<u8>,
    visible_cells: Vec<usize>,
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

    pub fn get_near_and_visible ( &self, cube_offsets: &[[f32; 3]], camera_position: &glm::Vec3 ) -> Vec<usize> {
        self.visible_cells.par_iter()
            .filter_map(|&idx| {
                // correct for vs scaling
                let offset = cube_offsets[idx].iter().map(|&x| (x as f32) / 100.0).collect::<Vec<_>>();

                let distance = glm::distance(
                    camera_position,
                    &glm::vec3(offset[0], offset[1], offset[2])
                );

                // arbitrary threshold
                if distance < 1.0 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
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
