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
from typing import List
from .objects import Link, Joint, Chain
import xml.etree.ElementTree as ET


class Urdf:
    root_xml: ET
    links: List[Link]
    joints: List[Joint]
    root_chains: List[Chain]

    def __init__(self, root_xml: ET, links: List[Link], joints: List[Joint]) -> None:
        self.root_xml = root_xml
        self.links = links
        self.joints = joints

    def init_from_file(urdf_file_path: str) -> Urdf:
        with open(urdf_file_path, "r") as f:
            return Urdf.init_from_string(f.read())

    def init_from_string(urdf_string: str) -> Urdf:
        urdf_root = ET.fromstring(urdf_string)
        links: List[Link] = [
            Link.from_xml(child) for child in urdf_root if child.tag == "link"
        ]

        joints: List[Joint] = [
            Joint.from_xml(child, links) for child in urdf_root if child.tag == "joint"
        ]

        urdf = Urdf(urdf_root, links, joints)
        urdf._make_chains()

        return urdf

    def print_tree(self):
        for i, chain in enumerate(self.root_chains):
            print("Root Chain ", i)
            chain.print()

    def get_all_chains(self):
        chains = self.root_chains.copy()
        for chain in self.root_chains:
            chains += chain.get_chains()
        return chains

    def merge_fixed_joints(
        self, merge_eef_chains: bool = False, except_eef_name: str = ""
    ):
        delete_list: List[Chain] = []
        add_list: List[Chain] = []
        for chain in self.root_chains:
            result = chain.merge_fixed_joints(merge_eef_chains, except_eef_name)
            if not result is None:
                delete_list.append(chain)
                add_list += result
        for element in delete_list:
            self.root_chains.remove(element)
        for element in add_list:
            self.root_chains.append(element)

    def _make_chains(self):
        root_link_names: List[str] = list(
            set([l.name for l in self.links])
            - set([joint.child.name for joint in self.joints])
        )

        root_links: List[Link] = [
            link for link in self.links if link.name in root_link_names
        ]

        self.root_chains = [
            Chain.create(root_links[0], joint) for joint in root_links[0].child_joints
        ]

    def update(self, keep_links: List[Link], keep_joints: List[Joint]):
        self.links = [l for l in self.links if l in keep_links]
        self.joints = [j for j in self.joints if j in keep_joints]

        to_remove = []
        for child in self.root_xml:
            if child.tag == "link":
                if not child.attrib["name"] in [l.name for l in self.links]:
                    to_remove.append(child)
            if child.tag == "joint":
                if not child.attrib["name"] in [j.name for j in self.joints]:
                    to_remove.append(child)

        for child in to_remove:
            # print(f"removing item {child.attrib['name']}")
            self.root_xml.remove(child)
