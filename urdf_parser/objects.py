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
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from xml.etree.ElementTree import Element
from .utils import (
    recurse_subtree,
    get_subtree,
    rotMat,
    homogMat,
    inertia_matrix_from_vec,
    Xcross,
    inertia_vec_from_matrix,
    parse_inertia_tree,
    parse_xyzrpy,
    xyzrpy_to_homo,
    axis_angle_to_matrix,
    normalize,
)
import numpy as np


@dataclass
class Link:
    xml: Element
    name: str
    mesh_file: str = None
    mass: float = None
    com: np.ndarray = None
    inertias: np.ndarray = None
    motor_inertia: float = 0.0
    child_joints: List[Joint] = field(default_factory=list)

    def from_xml(xml: Element) -> Link:
        l = Link(
            xml=xml,
            name=xml.attrib["name"],
        )
        try:
            l.mass = float(recurse_subtree(xml, ["inertial", "mass"]).attrib["value"])
            l.inertias = parse_inertia_tree(
                recurse_subtree(xml, ["inertial", "inertia"])
            )
        except AttributeError:
            l.mass = 0.0
            l.inertias = np.zeros(6)
        try:
            l.com = parse_xyzrpy(recurse_subtree(xml, ["inertial", "origin"]))
        except AttributeError:
            l.com = np.array([0, 0, 0, 0, 0, 0])

        try:
            l.mesh_file = recurse_subtree(xml, ["visual", "geometry", "mesh"]).attrib[
                "filename"
            ]
        except AttributeError:
            l.mesh_file = ""

        return l


class JointLimits:
    effort: float
    velocity: float
    upper: float
    lower: float

    @staticmethod
    def zero() -> JointLimits:
        joint_limits = JointLimits()
        joint_limits.upper = 0.0
        joint_limits.lower = 0.0
        joint_limits.velocity = 0.0
        joint_limits.effort = 0.0
        return joint_limits

    @staticmethod
    def from_xml(xml: Element) -> JointLimits:
        joint_limits = JointLimits()

        # get joint limits
        try:
            limits = get_subtree(xml, "limit")
        except AttributeError:
            joint_limits.effort = float("inf")
            joint_limits.velocity = float("inf")
            joint_limits.upper = float("inf")
            joint_limits.lower = float("-inf")
            return joint_limits

        try:
            joint_limits.effort = abs(float(limits.attrib["effort"]))
        except AttributeError:
            joint_limits.effort = float("inf")
        try:
            joint_limits.velocity = abs(float(limits.attrib["velocity"]))
        except AttributeError:
            joint_limits.velocity = float("inf")
        try:
            joint_limits.upper = float(limits.attrib["upper"])
        except (AttributeError, KeyError):
            joint_limits.upper = float("inf")
        try:
            joint_limits.lower = float(limits.attrib["lower"])
        except (AttributeError, KeyError):
            joint_limits.lower = float("-inf")

        if joint_limits.upper < joint_limits.lower:
            tmp_lower_value = joint_limits.lower
            joint_limits.lower = joint_limits.upper
            joint_limits.upper = tmp_lower_value

        return joint_limits


@dataclass
class Joint:
    xml: Element
    name: str
    type: str
    parent: Link
    child: Link
    origin: np.ndarray
    axis: np.ndarray
    limits: JointLimits
    static_friction: float = 0.0
    dynamic_friction: float = 0.0
    gear_ratio: float = 1.0
    rototraslation: np.ndarray = np.identity(4)

    def from_xml(xml: Element, links: List[Link]) -> Joint:
        parent_name = get_subtree(xml, "parent").attrib["link"]
        child_name = get_subtree(xml, "child").attrib["link"]
        parent = None
        child = None
        for link in links:
            if link.name == parent_name:
                parent = link
            elif link.name == child_name:
                child = link
        if parent is None:
            print("Wrong parent link of " + xml.attrib["name"] + " joint.")
        if child is None:
            print("Wrong parent link of " + xml.attrib["name"] + " joint.")

        try:
            origin = parse_xyzrpy(get_subtree(xml, "origin"))
        except AttributeError:
            origin = np.zeros((6))

        try:
            axis = parse_xyzrpy(get_subtree(xml, "axis"), False)
            axis = normalize(axis)
        except (AttributeError, ZeroDivisionError) as e:
            axis = np.array([1, 0, 0])

        j = Joint(
            xml=xml,
            name=xml.attrib["name"],
            type=xml.attrib["type"],
            origin=origin,
            parent=parent,
            child=child,
            axis=axis,
            limits=JointLimits.from_xml(xml)
            if not xml.attrib["type"] == "fixed"
            else JointLimits.zero(),
        )
        if not parent is None:
            parent.child_joints.append(j)
        return j

    @property
    def base_tf(self):
        return xyzrpy_to_homo(self.origin)

    def set_value(self, value: float) -> bool:
        val = self.clamp(value)
        if self.type == "prismatic":
            R = rotMat(np.zeros(3))
            self.rototraslation = homogMat(R, self.axis * val)
        elif self.type == "revolute" or self.type == "continuous":
            R = axis_angle_to_matrix(self.axis, value)
            self.rototraslation = homogMat(R, np.zeros(3))
        return val == value

    @property
    def tf(self):
        return self.base_tf @ self.rototraslation

    def clamp(self, value: float) -> float:
        if value < self.limits.lower:
            return self.limits.lower
        if value > self.limits.upper:
            return self.limits.upper

        return value


@dataclass
class Chain:
    name: str
    links: List[Link]
    joints: List[Joint]
    parent: Chain = None
    subchains: List[Chain] = field(default_factory=list)

    def create(link: Link, joint: Joint, parent: Chain = None) -> Chain:
        links = [link]
        joints = [joint]
        j = joint
        l = j.child
        while True:
            l = j.child
            links.append(l)
            if not len(l.child_joints) == 1:
                break
            j = l.child_joints[0]
            joints.append(j)

        last_link = j.child
        chain = Chain(
            name=joints[0].name,
            links=links,
            joints=joints,
            parent=parent,
        )

        chain.subchains = [
            Chain.create(last_link, joint, chain)
            for joint in last_link.child_joints
            if not last_link is None
        ]

        return chain

    def print(self, indentation: str = "") -> None:
        all_chains = self.get_chains()
        chain_print = self.links[0].name
        for link in self.links[1:]:
            chain_print += " -> " + link.name
        print(indentation + chain_print)
        if len(self.subchains) > 0:
            indentation += "  "
            print(indentation + "Subchains: ")
            i = 1
            for chain_list in all_chains:
                for chain in chain_list:
                    if type(chain) == Chain:
                        print(indentation + "- Subchain ", i)
                        chain.print(indentation + "  ")
                        i += 1

    def merge_fixed_joints(
        self, merge_eef_chains: bool = False, except_eef_names: List[str] = []
    ) -> List[Chain]:
        for joint in reversed(self.joints):
            condition = joint.type == "fixed"
            if not merge_eef_chains:
                # last fixed joint of the last chains is not merged (usually the end effector)
                condition *= len(joint.child.child_joints) != 0
            else:
                # all the end effector are merged except the one specified
                condition *= not joint.child.name in except_eef_names
            if condition:
                # if both links are without mass merge the dynamics is useless
                if joint.parent.mass != 0 or joint.child.mass != 0:
                    Chain.merge_dynamics(joint)
                Chain.merge_kinematics(joint)
                # Move the child joints back
                for j in joint.child.child_joints:
                    # Point to the new parent link
                    j.parent = joint.parent
                    # Add to the parent the new child joints
                    joint.parent.child_joints.append(j)

                # remove to the parent link the pointer to the removed joint
                joint.parent.child_joints.remove(joint)
                # remove from the lists the removed links and joints
                self.joints.remove(joint)
                self.links.remove(joint.child)
                for chain in self.subchains:
                    if joint.child in chain.links:
                        chain.links.remove(joint.child)
                        chain.links.insert(0, joint.parent)

        # CASE WHERE THE CHAIN HAS TO BE REFACTORED
        if len(self.links) == 1 and len(self.joints) == 0:
            for chain in self.subchains:
                chain.parent = None

            new_root_chains: List[Chain] = []
            for chain in self.subchains:
                result = chain.merge_fixed_joints(merge_eef_chains, except_eef_names)
                if not result is None:
                    new_root_chains += result
                else:
                    new_root_chains += [chain]
            return new_root_chains

        # NO UPPER LEVEL REFACTOR NEEDED
        delete_list: List[Chain] = []
        add_list: List[Chain] = []
        for chain in self.subchains:
            result = chain.merge_fixed_joints(merge_eef_chains, except_eef_names)
            if not result is None:
                delete_list.append(chain)
                for new_root_chain in result:
                    add_list.append(new_root_chain)
        for element in add_list:
            self.subchains.append(element)
        for element in delete_list:
            self.subchains.remove(element)

        if len(self.subchains) == 1:
            self.joints += chain.joints
            self.links += chain.links[1:]
            self.subchains = []
        return None

    def merge_dynamics(joint: Joint) -> None:
        # the dynamics coming from the link is moved to the previous link
        # IMPORTANT: rpy from center of mass position not considered (like if it is always rpz = 000)
        r_com_parent = joint.parent.com[:3]
        r_com_child = joint.child.com[:3]
        R = rotMat(np.array(joint.origin[3:]))
        H = homogMat(R, np.array(joint.origin[:3]))
        r_com_child_parent = np.transpose(
            H @ np.transpose(np.hstack((r_com_child, 1)))
        )[0:3]
        r_com_tot = (
            (r_com_parent * joint.parent.mass) + (r_com_child_parent * joint.child.mass)
        ) / (joint.parent.mass + joint.child.mass)
        joint.parent.com = np.array([*r_com_tot, 0, 0, 0])  # merged center of mass

        inertia1 = inertia_matrix_from_vec(np.array(joint.parent.inertias))
        inertia2 = inertia_matrix_from_vec(np.array(joint.child.inertias))

        delta_r_parent = (
            r_com_tot - r_com_parent
        )  # difference between r_com_1 and r_com_tot
        delta_r_child = r_com_tot - r_com_child_parent
        inertia_parent_com = inertia1 + joint.parent.mass * np.transpose(
            Xcross(delta_r_parent)
        ) @ Xcross(delta_r_parent)
        inertia_child_parent = np.transpose(R) @ inertia2 @ R
        inertia_child_com = inertia_child_parent + joint.child.mass * np.transpose(
            Xcross(delta_r_child)
        ) @ Xcross(delta_r_child)
        inertia_com_tot = inertia_parent_com + inertia_child_com

        joint.parent.inertias = inertia_vec_from_matrix(
            inertia_com_tot
        )  # merged inertias

        joint.parent.mass = joint.parent.mass + joint.child.mass  # merged mass

    def merge_kinematics(joint: Joint) -> None:
        # the kinematics coming from the joint is moved to all the joints next to it
        for next in joint.child.child_joints:
            H_tot = xyzrpy_to_homo(joint.origin) @ xyzrpy_to_homo(
                next.origin
            )  # compute full rototranslation

            xyz = H_tot[:3, 3]

            roll = np.arctan2(H_tot[2, 1], H_tot[2, 2])
            pitch = np.arctan2(
                -H_tot[2, 0], np.sqrt(H_tot[2, 1] ** 2 + H_tot[2, 2] ** 2)
            )
            yaw = np.arctan2(H_tot[1, 0], H_tot[0, 0])
            next.origin[:3] = xyz
            next.origin[3:] = np.array([roll, pitch, yaw])

    def get_chains(self) -> None:
        return [self.subchains] + [chain.get_chains() for chain in self.subchains]

    def get_root(self) -> Link:
        if self.parent is None:
            link = self.links[0]
        else:
            link = self.parent.get_root()
        return link

    def get_tf(self, link: Link) -> List[Tuple[Joint, np.ndarray]]:
        tfs: List[Tuple[Joint, np.ndarray]] = []
        if link in self.links:
            if not self.parent is None:
                tfs += self.parent.get_tf(self.parent.links[-1])
            tfs.append((self.joints[0], self.joints[0].tf))
            if link != self.links[0]:
                for l in self.links[1:]:
                    if l == link:
                        break
                    if not l is self.links[0]:
                        j = l.child_joints[0]
                        tfs.append((j, j.tf))

        return tfs

    @property
    def active_joints(self):
        return [j for j in self.joints if not j.type == "fixed"]
