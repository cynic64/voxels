extern crate rand;
extern crate rayon;
extern crate bytecount;
use self::rayon::prelude::*;

pub struct CellA {
    pub cells: Vec<u8>,
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
        let max_age = 2;

        Self {
            cells,
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

    pub fn set_xyz ( &mut self, x: usize, y: usize, z: usize, new_state: u8 ) {
        let idx = (z * self.width * self.height) + (y * self.width) + x;
        self.cells[idx] = new_state;
    }

    pub fn randomize ( &mut self ) {
        let cells = (0 .. self.width * self.height * self.length)
            .map(|x| {
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
