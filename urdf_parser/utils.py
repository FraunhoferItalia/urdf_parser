#
# Copyright 2022-2024 Fraunhofer Italia Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
from typing import List
from xml.etree.ElementTree import Element
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def axis_angle_to_matrix(vec: np.ndarray, angle: float) -> np.ndarray:
    if vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2 != 1:
        vec = normalize(vec)
    return (
        np.identity(3) * np.cos(angle)
        + Xcross(vec) * np.sin(angle)
        + (1 - np.cos(angle)) * np.outer(vec, vec)
    )


def get_subtree(tree: Element, subtree_name: str) -> Element:
    for child in tree:
        if child.tag == subtree_name:
            return child


def recurse_subtree(tree: Element, subtree_names: List[str]) -> Element:
    next_element = copy.deepcopy(tree)
    for i in range(len(subtree_names)):
        next_element = get_subtree(next_element, subtree_names[i])
        if next_element is None:
            break

    return next_element


def parse_xyzrpy(tree: Element, rpy: bool = True) -> List[float]:
    result = tree.attrib["xyz"].split()
    if rpy:
        try:
            result += tree.attrib["rpy"].split()
        except KeyError:
            result += ["0.0", "0.0", "0.0"]
    return np.array(result).astype(np.float32)


def xyzrpy_to_homo(xyzrpy: np.ndarray):
    H = np.eye(4)
    H[:3, :3] = R.from_euler("xyz", xyzrpy[3:]).as_matrix()
    H[:3, 3] = np.array(xyzrpy[:3])
    return H


def homo_to_xyzrpy(xyzrpy: np.ndarray):
    # NOT IMPLEMENTED
    return np.zeros([6, 1])


def parse_inertia_tree(tree: Element) -> List[float]:
    return np.array(
        [
            float(tree.attrib["ixx"]),
            float(tree.attrib["iyy"]),
            float(tree.attrib["izz"]),
            float(tree.attrib["ixy"]),
            float(tree.attrib["ixz"]),
            float(tree.attrib["iyz"]),
        ]
    )


def inertia_matrix_from_vec(inertia_vect):  #  Ixx, Iyy, Izz, Ixy, Ixz, Iyz
    # function to build inertia matrix from inertia vector
    return np.array(
        [
            [inertia_vect.item(0), inertia_vect.item(3), inertia_vect.item(4)],
            [inertia_vect.item(3), inertia_vect.item(1), inertia_vect.item(5)],
            [inertia_vect.item(4), inertia_vect.item(5), inertia_vect.item(2)],
        ]
    )


def inertia_vec_from_matrix(inertiaVect):  #  Ixx, Iyy, Izz, Ixy, Ixz, Iyz
    # function to build inertia matrix from inertia vector
    return np.array(
        [
            inertiaVect[0][0],
            inertiaVect[1][1],
            inertiaVect[2][2],
            inertiaVect[0][1],
            inertiaVect[0][2],
            inertiaVect[1][2],
        ]
    )


def newton_euler_urdf(q, dq, ddq, robot_parameters):
    k = robot_parameters.nDof + 1  # number of arms = number of joints + 1
    jointType = robot_parameters.jType
    g = robot_parameters.g

    m = robot_parameters.DynPar.m
    I = np.zeros((k, 3, 3))
    for i in range(k):
        I[i, :, :] = inertia_matrix_from_vec(np.array(robot_parameters.DynPar.I)[i, :])

    r = np.transpose(np.array(robot_parameters.DynPar.r))

    xyz = np.transpose(np.array(robot_parameters.KinPar.xyz))
    rpy = np.transpose(np.array(robot_parameters.KinPar.rpy))

    kr = robot_parameters.DynPar.kr
    Im = robot_parameters.DynPar.Im
    Fv = robot_parameters.DynPar.Fv
    Fc = robot_parameters.DynPar.Fc

    # initialize parameters
    d = np.zeros(k)
    theta = np.zeros(k)

    omega = np.zeros((3, k))
    omegaDot = np.zeros((3, k))
    a_j = np.zeros((3, k))
    a_c = np.zeros((3, k))
    R = np.zeros((k + 1, 3, 3))  # from i to i-1
    R_T = np.zeros((k, 3, 3))  # from i-1 to i
    z = np.zeros((3, k))
    u = np.zeros(k)
    z0 = np.array([[0], [0], [1]])
    f = np.zeros((3, k + 1))  # forces (k+1 for end effector forces)
    mu = np.zeros((3, k + 1))  # torques (k+1 for end effector torques)

    r_link = np.zeros((3, k))

    for i in range(k):
        if i != k - 1:
            R[i, :, :] = rotMat(rpy[:, i]).dot(RzMat(q[i])[0:3, 0:3])
            R_T[i, :, :] = (R[i, :, :]).transpose()

            z[:, [i]] = R_T[i, :, :].dot(z0)

            r_link[:, [i]] = xyz[:, [i]]

        else:
            R[i, :, :] = rotMat(rpy[:, i])
            R_T[i, :, :] = (R[i, :, :]).transpose()

            z[:, [i]] = R_T[i, :, :].dot(z0)

            r_link[:, [i]] = xyz[:, [i]]

    B = np.eye(3)

    # forward recursion

    for i in range(k):
        # initialization
        if i == 0:
            z[:, [i]] = z0
            a_j[:, [i]] = -B.transpose().dot(np.array([[g[0]], [g[1]], [g[2]]]))

        else:
            if jointType[i - 1] == 1:  # revolute joint
                omega[:, [i]] = (
                    R_T[i - 1, :, :].dot(omega[:, [i - 1]]) + dq.item(i - 1) * z0
                )  # dq[i-1] because it starts from i = 1

                omegaDot[:, [i]] = (
                    R_T[i - 1, :, :].dot(omegaDot[:, [i - 1]])
                    + Xcross(R_T[i - 1, :, :].dot(omega[:, [i - 1]])).dot(
                        dq.item(i - 1) * z0
                    )
                    + ddq.item(i - 1) * z0
                )
                a_j[:, [i]] = R_T[i - 1, :, :].dot(
                    a_j[:, [i - 1]]
                    + Xcross(omegaDot[:, i - 1]).dot(r_link[:, [i - 1]])
                    + Xcross(omega[:, i - 1]).dot(
                        Xcross(omega[:, i - 1]).dot(r_link[:, [i - 1]])
                    )
                )
            elif jointType[i - 1] == 0:  # prismatic joint
                # omega[:,[i]] = R_T[i-1,:,:].dot(omega[:,[i-1]])
                # omegaDot[:,[i]] = R_T[i-1,:,:].dot(omegaDot[:,[i-1]])
                # a_j[:,[i]] = R_T[i-1,:,:].dot(a_j[:,[i-1]]) + Xcross(omegaDot[:,i]).dot(r_link[:,[i]]) + Xcross(omega[:,i]).dot(Xcross(omega[:,i]).dot(r_link[:,[i]])) + 2*Xcross(omega[:,i]).dot(dq.item(i-1)*z[:,[i-1]]) + ddq.item(i-1)*z[:,[i-1]]
                raise Exception("Prismatic joint not implemented")
        a_c[:, [i]] = (
            a_j[:, [i]]
            + Xcross(omegaDot[:, i]).dot(r[:, [i]])
            + Xcross(omega[:, i]).dot(Xcross(omega[:, i]).dot(r[:, [i]]))
        )

    # backward recursion

    for i in range(k - 1, 0, -1):
        if i == k - 1:  # force and torques at the end effector
            f[:, i + 1] = np.zeros(3)
            mu[:, i + 1] = np.zeros(3)
            R[i + 1, :, :] = np.eye(3)

        Fi = m[i] * a_c[:, [i]]
        f[:, [i]] = R[i, :, :].dot(f[:, [i + 1]]) + Fi

        mu[:, [i]] = (
            R[i, :, :].dot(mu[:, [i + 1]])
            + (Xcross(R[i, :, :].dot(f[:, [i + 1]])).dot(r[:, [i]] - r_link[:, [i]]))
            - (Xcross(f[:, [i]]).dot(r[:, [i]]))
            + (I[i, :, :].dot(omegaDot[:, [i]]))
            + (Xcross(omega[:, i]).dot(I[i, :, :].dot(omega[:, [i]])))
        )
        if jointType[i - 1] == 1:  # revolute joint
            u[i] = (
                (mu[:, [i]]).transpose().dot(z0)
                + Fv[i - 1] * dq.item(i - 1)
                + Fc[i - 1] * np.sin(np.arctan(1000 * dq.item(i - 1)))
                + ddq.item(i - 1) * kr[i - 1] ** 2 * Im[i - 1]
            )  # + friq(dq[i]) + sigma[i]**2*Im[i,:,:]*ddq[i]
        elif jointType[i - 1] == 0:  # prismatic joint
            # u[i] = (f[:,[i]]).transpose().dot(z0) + Fv[i-1]*dq.item(i-1)+Fc[i-1]*np.sin(np.arctan(1000*dq.item(i-1))) + ddq.item(i-1)*kr[i-1]**2*Im[i-1] # + friq(dq[i]) + sigma[i]**2*Im[i,:,:]*ddq[i]
            raise Exception("Prismatic joint not implemented")

    u_out = u[1:]
    return u_out


# Rotation and translation matrices
def RxMat(x):
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), -np.sin(x), 0],
            [0, np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ]
    )
    return Rx


def RzMat(x):
    Rz = np.array(
        [
            [np.cos(x), -np.sin(x), 0, 0],
            [np.sin(x), np.cos(x), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return Rz


def TxMat(x):
    Tx = np.array([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return Tx


def TzMat(x):
    Tz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, x], [0, 0, 0, 1]])
    return Tz


def Xcross(x):
    """compute 3x3 cross matrix.
    Args:
        x (np.array 3x1 or 1x3): input vector

    Returns:
        3x3 np.array
    """
    XMat = np.array(
        [
            [0, -x.item(2), x.item(1)],
            [x.item(2), 0, -x.item(0)],
            [-x.item(1), x.item(0), 0],
        ]
    )
    return XMat


def rotMat(angles) -> np.ndarray:
    x = angles.item(0)  # roll
    y = angles.item(1)  # pitch
    z = angles.item(2)  # yaw

    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = np.dot(np.dot(Rz, Ry), Rx)
    return R


def maxEigComputation(robot_parameters, MC_points):
    """
    Compute max eigenvalue of the inertia for a robot manipulator
    by performing a small Monte Carlo run.
    It also computes the average inertia value
    """

    robot_parameters_no_g = copy.deepcopy(
        robot_parameters
    )  # copied because g is set to null for B_mat computation
    robot_parameters_no_g.g = [0, 0, 0]  # removed gravity

    n = robot_parameters_no_g.nDof
    dq_zero = np.zeros((1, n))
    eye_mat = np.eye(n)  # eye matrix

    max_eig_list = np.zeros(MC_points)
    inertia_vect = np.zeros((MC_points, n))

    """ print("testing M generation")

    B_mat = np.zeros((n,n)) # inertia (also commonly called M)
    q = np.array([1.1755, 1.32677, -1.39991, 0.960008])
    for j in range(n):

        B_mat[:,j] = np.transpose(newton_euler_urdf(q, dq_zero, eye_mat[j,:], robot_parameters_no_g))
    
    print("printing inertia")
    print(B_mat) """

    for i in range(MC_points):
        q = 2 * np.pi * np.random.uniform(-0.5, 0.5, (n))

        B_mat = np.zeros((n, n))  # inertia (also commonly called M)

        for j in range(n):
            B_mat[:, j] = np.transpose(
                newton_euler_urdf(q, dq_zero, eye_mat[j, :], robot_parameters_no_g)
            )

        inertia_vect[i, :] = np.diag(B_mat)

        eig_values, _ = np.linalg.eig(B_mat)
        max_eig_list[i] = np.max(eig_values)

    avg_inertia = np.average(inertia_vect, axis=0)
    max_eig_value = np.max(max_eig_list)

    return max_eig_value, avg_inertia


def homogMat(R, v):
    H = np.array(
        [
            [R[0][0], R[0][1], R[0][2], v.item(0)],
            [R[1][0], R[1][1], R[1][2], v.item(1)],
            [R[2][0], R[2][1], R[2][2], v.item(2)],
            [0, 0, 0, 1],
        ]
    )
    return H
